use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::db::{CheckOptions, CompactOptions, Database, DatabaseOpenOptions, RecordKey};
use crate::error::{Error, ErrorCode, Result};
use crate::json::{
    object_from_pairs, parse, JsonBuilder, JsonInputKind, JsonOperationContext, JsonOutputSchema,
    JsonReservation, JsonSchema, JsonValue,
};
use crate::metadata::MetadataBoostConfig;
use crate::model::{
    CollectionConfig, DistanceMetric, OnConflict, QueryEmbeddingInput, SearchWeights,
    VectorIndexConfig,
};
use crate::profile::{DatabaseConfig, Durability, ResourceProfile};
use crate::storage::FORMAT_VERSION;

pub fn run(args: Vec<String>) -> Result<String> {
    run_with_exit_code(args).map(|output| output.0)
}

pub fn run_with_exit_code(args: Vec<String>) -> Result<(String, i32)> {
    let started = Instant::now();
    let mut args = Args::new(args);
    let _program = args.next();
    let Some(command) = args.next() else {
        return Err(usage("missing command"));
    };
    let result = run_command(command.as_str(), args)?;
    let exit_code = success_exit_code(&command, &result.value);
    Ok((
        success_output(&command, result, started.elapsed().as_millis())?,
        exit_code,
    ))
}

pub fn error_output(command: Option<&str>, err: &Error) -> String {
    cli_envelope(
        command.unwrap_or("unknown"),
        false,
        JsonValue::object(),
        Some(err.to_json()),
        0,
    )
}

pub fn human_output(output: &str) -> String {
    let Ok(envelope) = parse(output) else {
        return output.to_owned();
    };
    let command = envelope
        .get("command")
        .and_then(JsonValue::as_str)
        .unwrap_or("coffer");
    let mut summary = format!("{command}\n");
    if let Some(result) = envelope.get("result").and_then(JsonValue::as_object) {
        for (key, value) in result {
            if key == "results" {
                if let Some(items) = value.as_array() {
                    for item in items {
                        summary.push_str(&format!(
                            "  {}  score={}\n    {}\n",
                            item.get("chunk_external_id")
                                .and_then(JsonValue::as_str)
                                .unwrap_or("?"),
                            item.get("final_score")
                                .map(JsonValue::to_compact_string)
                                .unwrap_or_default(),
                            item.get("snippet")
                                .and_then(JsonValue::as_str)
                                .unwrap_or("")
                        ));
                    }
                }
            } else {
                summary.push_str(&format!("  {key}: {}\n", value.to_compact_string()));
            }
        }
    }
    summary
}

struct CommandResult {
    value: JsonValue,
    retained: Option<(JsonOperationContext, JsonReservation, JsonOutputSchema)>,
}

impl CommandResult {
    fn plain(value: JsonValue) -> Self {
        Self {
            value,
            retained: None,
        }
    }

    fn retained(
        value: JsonValue,
        context: JsonOperationContext,
        work: JsonReservation,
        schema: JsonOutputSchema,
    ) -> Self {
        Self {
            value,
            retained: Some((context, work, schema)),
        }
    }
}

fn run_command(command: &str, mut args: Args) -> Result<CommandResult> {
    let result = match command {
        "init" => {
            let path = args.required("path")?;
            args.reject_remaining()?;
            Database::create(&path, DatabaseConfig::default())?;
            Ok(object_from_pairs([
                ("path", JsonValue::String(path)),
                ("format_version", JsonValue::Number(FORMAT_VERSION as f64)),
                ("created", JsonValue::Bool(true)),
            ]))
        }
        "create-collection" => {
            let path = args.required("path")?;
            let name = args.required("name")?;
            let embedding_dim = parse_embedding_dim_flag(&mut args)?;
            let distance_metric = parse_distance_metric_flag(&mut args)?;
            let embedding_model = args.flag_value("--embedding-model")?;
            let tokenizer = args
                .flag_value("--tokenizer")?
                .map(|name| crate::TokenizerConfig::from_name(&name))
                .transpose()?
                .unwrap_or_default();
            let vectors_normalized = args.has_flag("--vectors-normalized");
            args.reject_remaining()?;
            let db = Database::open(path)?;
            db.create_collection(CollectionConfig {
                tokenizer: tokenizer.clone(),
                name: name.clone(),
                embedding_dim,
                distance_metric,
                embedding_model: embedding_model.clone(),
                vectors_normalized,
                vector_index: VectorIndexConfig::Exact,
            })?;
            Ok(object_from_pairs([
                ("collection", JsonValue::String(name)),
                ("tokenizer", tokenizer.to_json()),
                ("embedding_dim", JsonValue::Number(embedding_dim as f64)),
                (
                    "distance_metric",
                    JsonValue::String(distance_metric.as_str().into()),
                ),
                ("embedding_model", opt_string(embedding_model)),
                ("vectors_normalized", JsonValue::Bool(vectors_normalized)),
            ]))
        }
        "list-collections" => {
            let path = args.required("path")?;
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            let context = db.json_context();
            let (value, work) = db.list_collections_json(&context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "collection-stats" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let include_index = args.has_flag("--include-index");
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            let context = db.json_context();
            let (value, work) = db.stats_json(Some(&collection), include_index, &context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "import" => {
            let path = args.required("path")?;
            let manifest = args
                .flag_value("--manifest")?
                .ok_or_else(|| usage("--manifest is required"))?;
            let durability = parse_durability_flag(&mut args)?;
            let on_conflict = args
                .flag_value("--on-conflict")?
                .map(|value| OnConflict::parse(&value))
                .transpose()?
                .unwrap_or(OnConflict::Error);
            args.reject_remaining()?;
            let db = if PathBuf::from(&path).exists() {
                open_database_with_durability(&path, durability)?
            } else {
                Database::create(
                    &path,
                    DatabaseConfig {
                        durability,
                        ..DatabaseConfig::default()
                    },
                )?
            };
            let context = db.json_context();
            let imported = db.import_manifest_measured_in(
                std::path::Path::new(&manifest),
                on_conflict,
                crate::db::ImportIndexBuildMode::DeferredFullCheckpoint,
                context.clone(),
            )?;
            let (result, work) =
                import_output_in(&db, &imported.summary, &manifest, durability, &context)?;
            drop(imported);
            return Ok(CommandResult::retained(
                result,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "import-embeddings" => {
            let path = args.required("path")?;
            let embeddings = args.required("embeddings.jsonl")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let durability = parse_durability_flag(&mut args)?;
            let on_conflict = args
                .flag_value("--on-conflict")?
                .map(|value| OnConflict::parse(&value))
                .transpose()?
                .unwrap_or(OnConflict::Replace);
            args.reject_remaining()?;
            let db = open_database_with_durability(&path, durability)?;
            let col = db.collection(&collection)?;
            let summary = db.import_embeddings_jsonl(
                col.collection_id(),
                PathBuf::from(embeddings).as_path(),
                on_conflict,
            )?;
            return retained_summary_output(&db, |builder| {
                let mut result = summary.build_json(builder)?;
                builder.insert_if_absent(result.as_object_mut().unwrap(), "durability", || {
                    builder.string(durability.as_str())
                })?;
                Ok(result)
            });
        }
        "search" => return search_command(args, false),
        "vector-search" => return search_command(args, true),
        "why-not" => return why_not_command(args),
        "get-document" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let include_chunks = args.has_flag("--include-chunks");
            let id = args.flag_value("--document-id")?;
            let external = if id.is_none() {
                Some(
                    args.flag_value("--external-id")?
                        .ok_or_else(|| usage("--document-id or --external-id is required"))?,
                )
            } else {
                None
            };
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            let col = db.collection(&collection)?;
            let key = if let Some(id) = id.as_deref() {
                RecordKey::Id(id)
            } else {
                RecordKey::ExternalId(external.as_deref().unwrap())
            };
            let context = db.json_context();
            let (value, work) =
                db.document_json_in_collection(col.collection_id(), key, include_chunks, &context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "get-chunk" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let id = args.flag_value("--chunk-id")?;
            let external = if id.is_none() {
                Some(
                    args.flag_value("--chunk-external-id")?
                        .ok_or_else(|| usage("--chunk-id or --chunk-external-id is required"))?,
                )
            } else {
                None
            };
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            let col = db.collection(&collection)?;
            let key = if let Some(id) = id.as_deref() {
                RecordKey::Id(id)
            } else {
                RecordKey::ExternalId(external.as_deref().unwrap())
            };
            let context = db.json_context();
            let (value, work) = db.chunk_json_in_collection(col.collection_id(), key, &context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "delete-document" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let db = Database::open(path)?;
            let col = db.collection(&collection)?;
            let document = if let Some(id) = args.flag_value("--document-id")? {
                col.get_document(&id)?
            } else {
                let external = args
                    .flag_value("--external-id")?
                    .ok_or_else(|| usage("--document-id or --external-id is required"))?;
                col.get_document_by_external_id(&external)?
            };
            let chunks = col.list_chunks(&document.document_id)?;
            let chunks_deleted = chunks.len();
            let embeddings_deleted = chunks
                .iter()
                .filter(|chunk| chunk.embedding_id.is_some())
                .count();
            args.reject_remaining()?;
            col.delete_document(&document.document_id)?;
            Ok(object_from_pairs([
                (
                    "document_id",
                    JsonValue::String(document.document_id.clone()),
                ),
                (
                    "external_id",
                    JsonValue::String(document.external_id.clone()),
                ),
                ("chunks_deleted", JsonValue::Number(chunks_deleted as f64)),
                (
                    "embeddings_deleted",
                    JsonValue::Number(embeddings_deleted as f64),
                ),
                (
                    "tombstone_id",
                    tombstone_result_id("document", &document.document_id),
                ),
                ("deleted", JsonValue::Bool(true)),
            ]))
        }
        "delete-chunk" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let db = Database::open(path)?;
            let col = db.collection(&collection)?;
            let chunk = if let Some(id) = args.flag_value("--chunk-id")? {
                col.get_chunk(&id)?
            } else {
                let external = args
                    .flag_value("--chunk-external-id")?
                    .ok_or_else(|| usage("--chunk-id or --chunk-external-id is required"))?;
                col.get_chunk_by_external_id(&external)?
            };
            let embedding_deleted = chunk.embedding_id.is_some();
            args.reject_remaining()?;
            col.delete_chunk(&chunk.chunk_id)?;
            Ok(object_from_pairs([
                ("chunk_id", JsonValue::String(chunk.chunk_id.clone())),
                (
                    "chunk_external_id",
                    JsonValue::String(chunk.chunk_external_id.clone()),
                ),
                ("embedding_deleted", JsonValue::Bool(embedding_deleted)),
                (
                    "tombstone_id",
                    tombstone_result_id("chunk", &chunk.chunk_id),
                ),
                ("deleted", JsonValue::Bool(true)),
            ]))
        }
        "drop-collection" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            args.reject_remaining()?;
            let db = Database::open(path)?;
            let col = db.collection(&collection)?;
            let collection_id = col.collection_id().to_owned();
            let stats = col.stats()?;
            db.drop_collection(&collection)?;
            Ok(object_from_pairs([
                ("collection", JsonValue::String(collection)),
                (
                    "documents_deleted",
                    JsonValue::Number(stats.document_count as f64),
                ),
                (
                    "chunks_deleted",
                    JsonValue::Number(stats.chunk_count as f64),
                ),
                (
                    "embeddings_deleted",
                    JsonValue::Number(stats.embedding_count as f64),
                ),
                (
                    "tombstone_id",
                    tombstone_result_id("collection", &collection_id),
                ),
                ("dropped", JsonValue::Bool(true)),
            ]))
        }
        "stats" => {
            let path = args.required("path")?;
            let include_index = args.has_flag("--include-index");
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            let context = db.json_context();
            let (value, work) = db.stats_json(None, include_index, &context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "check" => {
            let path = args.required("path")?;
            let _verify_indexes = args.has_flag("--verify-indexes");
            args.reject_remaining()?;
            let report = Database::check_path(
                path,
                CheckOptions {
                    verify_indexes: true,
                },
            )?;
            return retained_output_in(JsonOperationContext::standalone(), |builder| {
                report.build_json(builder)
            });
        }
        "rebuild-index" => {
            let path = args.required("path")?;
            let collection = args.flag_value("--collection")?;
            args.reject_remaining()?;
            let db = Database::open(path)?;
            let summary = db.rebuild_index(collection.as_deref())?;
            return retained_summary_output(&db, |builder| summary.build_json(builder));
        }
        "repair" => {
            let path = args.required("path")?;
            if !args.has_flag("--truncate-tail") {
                return Err(usage("repair requires --truncate-tail"));
            }
            args.reject_remaining()?;
            let summary = Database::repair_truncate_tail(path)?;
            return retained_output_in(JsonOperationContext::standalone(), |builder| {
                summary.build_json(builder)
            });
        }
        "snapshot" => {
            let path = args.required("path")?;
            let output_path = args.required("output-path")?;
            args.reject_remaining()?;
            let summary = Database::snapshot(path, output_path)?;
            return retained_output_in(JsonOperationContext::standalone(), |builder| {
                summary.build_json(builder)
            });
        }
        "compact" => {
            let path = args.required("path")?;
            let _rebuild_indexes = args.has_flag("--rebuild-indexes");
            args.reject_remaining()?;
            let db = Database::open(path)?;
            let summary = db.compact(CompactOptions {
                rebuild_indexes: true,
            })?;
            return retained_summary_output(&db, |builder| summary.build_json(builder));
        }
        "export" => {
            let path = args.required("path")?;
            let out = args.required("output-dir")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let format = args
                .flag_value("--format")?
                .unwrap_or_else(|| "jsonl".into());
            if format != "jsonl" {
                return Err(usage("--format must be jsonl")
                    .with_detail("field", "--format")
                    .with_detail("actual", &format)
                    .with_detail("expected", "jsonl"));
            }
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            let summary = db.export_collection_jsonl(&collection, &out)?;
            return retained_summary_output(&db, |builder| {
                let mut result = summary.build_json(builder)?;
                let obj = result.as_object_mut().unwrap();
                builder.insert_if_absent(obj, "output_dir", || builder.string(&out))?;
                builder.insert_if_absent(obj, "collection", || builder.string(&collection))?;
                builder.insert_if_absent(obj, "format", || builder.string(&format))?;
                Ok(result)
            });
        }
        "help" => Ok(help_json()),
        "profile-help" => Ok(profile_help_json()),
        "benchmark" => return benchmark_command(args),
        _ => Err(usage(format!("unknown command {command}"))
            .with_detail("field", "command")
            .with_detail("actual", command)
            .with_detail("reason", "unknown_command")),
    }?;
    args.reject_remaining()?;
    Ok(CommandResult::plain(result))
}

fn import_output_in(
    db: &Database,
    summary: &crate::import::ImportSummary,
    manifest: &str,
    durability: Durability,
    context: &JsonOperationContext,
) -> Result<(JsonValue, JsonReservation)> {
    db.admit_input_reclaiming(|| {
        build_output_in(context, |builder| {
            let mut result = summary.build_json(builder)?;
            let obj = result.as_object_mut().unwrap();
            builder.insert_if_absent(obj, "manifest", || builder.string(manifest))?;
            builder.insert_if_absent(obj, "durability", || builder.string(durability.as_str()))?;
            Ok(result)
        })
    })
}

fn build_output_in(
    context: &JsonOperationContext,
    build: impl FnOnce(&JsonBuilder<'_>) -> Result<JsonValue>,
) -> Result<(JsonValue, JsonReservation)> {
    let mut work = context.reservation();
    let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
    let result = build(&builder)?;
    Ok((result, work))
}

fn retained_summary_output(
    db: &Database,
    build: impl Fn(&JsonBuilder<'_>) -> Result<JsonValue>,
) -> Result<CommandResult> {
    let context = db.json_context();
    db.admit_input_reclaiming(|| retained_output_in(context.clone(), &build))
}

fn retained_output_in(
    context: JsonOperationContext,
    build: impl FnOnce(&JsonBuilder<'_>) -> Result<JsonValue>,
) -> Result<CommandResult> {
    let (value, work) = build_output_in(&context, build)?;
    Ok(CommandResult::retained(
        value,
        context,
        work,
        JsonOutputSchema::Value(JsonInputKind::StoredJson),
    ))
}

fn success_output(command: &str, result: CommandResult, duration_ms: u128) -> Result<String> {
    let mut retained = result.retained;
    let Some((context, work, schema)) = &mut retained else {
        return Ok(cli_envelope(command, true, result.value, None, duration_ms));
    };
    let builder = JsonBuilder::budgeted(work, schema.kind());
    let warnings = match result.value.get("warnings").and_then(JsonValue::as_array) {
        Some(items) => builder.array(
            items
                .iter()
                .map(|warning| public_warning_in(warning, &builder)),
        )?,
        None => JsonValue::Array(Vec::new()),
    };
    let envelope = build_cli_envelope(
        command,
        true,
        result.value,
        warnings,
        None,
        duration_ms,
        &builder,
    )?;
    let (text, output_work) = envelope.pretty_retained_in(*schema, context, work)?;
    drop(envelope);
    drop(retained);
    drop(output_work);
    Ok(text)
}

fn success_exit_code(command: &str, result: &JsonValue) -> i32 {
    if command != "check" {
        return 0;
    }
    if result
        .get("issues")
        .and_then(JsonValue::as_array)
        .is_some_and(|issues| {
            !issues.is_empty()
                && issues.iter().all(|issue| {
                    issue.get("code").and_then(JsonValue::as_str)
                        == Some("embedding_model_mismatch")
                })
        })
    {
        return 2;
    }
    match result.get("status").and_then(JsonValue::as_str) {
        Some("ok") | None => 0,
        Some(_) => 3,
    }
}

fn cli_envelope(
    command: &str,
    ok: bool,
    result: JsonValue,
    error: Option<JsonValue>,
    duration_ms: u128,
) -> String {
    cli_envelope_value(command, ok, result, error, duration_ms).to_pretty_string() + "\n"
}

fn cli_envelope_value(
    command: &str,
    ok: bool,
    result: JsonValue,
    error: Option<JsonValue>,
    duration_ms: u128,
) -> JsonValue {
    let warnings = result
        .get("warnings")
        .and_then(JsonValue::as_array)
        .map(|items| JsonValue::Array(items.iter().map(public_warning).collect()))
        .unwrap_or_else(|| JsonValue::Array(Vec::new()));
    build_cli_envelope(
        command,
        ok,
        result,
        warnings,
        error,
        duration_ms,
        &JsonBuilder::host(),
    )
    .expect("host envelope construction is infallible")
}

fn build_cli_envelope(
    command: &str,
    ok: bool,
    result: JsonValue,
    warnings: JsonValue,
    error: Option<JsonValue>,
    duration_ms: u128,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    let mut envelope = builder.object([
        ("schema_version", builder.string("0.1")?),
        ("command", builder.string(command)?),
        ("ok", JsonValue::Bool(ok)),
        ("warnings", warnings),
        ("duration_ms", JsonValue::Number(duration_ms as f64)),
        ("result", result),
    ])?;
    if let Some(error) = error {
        builder.insert_if_absent(
            envelope
                .as_object_mut()
                .expect("CLI envelope is always an object"),
            "error",
            || Ok(error),
        )?;
    }
    Ok(envelope)
}

fn public_warning(warning: &JsonValue) -> JsonValue {
    public_warning_in(warning, &JsonBuilder::host())
        .expect("host warning construction is infallible")
}

fn public_warning_in(warning: &JsonValue, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    if warning.as_object().is_some() {
        let mut value = builder.copy_validated_value(warning)?;
        let object = value.as_object_mut().expect("warning object copied above");
        if !object.contains_key("details") {
            let mut details = builder.object([])?;
            for key in ["field", "artifact_kind", "reason"] {
                if let Some(value) = object.get(key) {
                    builder.insert_if_absent(details.as_object_mut().unwrap(), key, || {
                        builder.copy_validated_value(value)
                    })?;
                }
            }
            builder.insert_if_absent(object, "details", || Ok(details))?;
        }
        builder.insert_if_absent(object, "severity", || builder.string("warning"))?;
        builder.insert_if_absent(object, "path", || Ok(JsonValue::Null))?;
        return Ok(value);
    }
    builder.object([
        ("code", builder.string("warning")?),
        ("severity", builder.string("warning")?),
        ("path", JsonValue::Null),
        (
            "message",
            builder.string(warning.as_str().unwrap_or("warning"))?,
        ),
        ("details", JsonValue::object()),
    ])
}

fn opt_string(value: Option<String>) -> JsonValue {
    value.map(JsonValue::String).unwrap_or(JsonValue::Null)
}

#[cfg(test)]
fn insert_object_field(value: &mut JsonValue, key: &str, item: JsonValue) {
    value
        .as_object_mut()
        .expect("CLI command result is always a JSON object")
        .insert(key.into(), item);
}

fn tombstone_result_id(target_type: &str, target_id: &str) -> JsonValue {
    JsonValue::String(format!("tombstone:{target_type}:{target_id}"))
}

#[derive(Clone, Copy)]
enum ParsedProfile {
    Help,
    Profile(ResourceProfile),
}

fn parse_profile_flag(args: &mut Args) -> Result<Option<ParsedProfile>> {
    args.flag_value("--profile")?
        .map(|value| {
            if value == "help" {
                Ok(ParsedProfile::Help)
            } else {
                ResourceProfile::parse(&value)
                    .map(ParsedProfile::Profile)
                    .ok_or_else(|| {
                        usage("invalid --profile")
                            .with_detail("field", "--profile")
                            .with_detail("actual", &value)
                            .with_detail("expected", "expanded|balanced|constrained|help")
                    })
            }
        })
        .transpose()
}

fn parse_durability_flag(args: &mut Args) -> Result<Durability> {
    args.flag_value("--durability")?
        .map(|value| {
            Durability::parse(&value).ok_or_else(|| {
                usage("invalid --durability")
                    .with_detail("field", "--durability")
                    .with_detail("actual", &value)
                    .with_detail("expected", "commit|fast")
            })
        })
        .transpose()
        .map(|value| value.unwrap_or(Durability::Commit))
}

fn parse_top_k_flag(args: &mut Args) -> Result<usize> {
    let value = args
        .flag_value("--top-k")?
        .unwrap_or_else(|| "10".to_owned());
    value.parse::<usize>().map_err(|_| {
        usage("--top-k must be an integer")
            .with_detail("field", "--top-k")
            .with_detail("actual", &value)
            .with_detail("expected", "positive_integer")
    })
}

fn parse_embedding_dim_flag(args: &mut Args) -> Result<usize> {
    let value = args
        .flag_value("--dim")?
        .ok_or_else(|| usage("--dim is required"))?;
    let dim = value
        .parse::<usize>()
        .map_err(|_| invalid_embedding_dim_usage(&value))?;
    if dim == 0 {
        return Err(invalid_embedding_dim_usage(&value));
    }
    Ok(dim)
}

fn invalid_embedding_dim_usage(value: &str) -> Error {
    usage("--dim must be a positive integer")
        .with_detail("field", "--dim")
        .with_detail("actual", value)
        .with_detail("expected", "positive_integer")
}

fn parse_distance_metric_flag(args: &mut Args) -> Result<DistanceMetric> {
    let value = args
        .flag_value("--metric")?
        .unwrap_or_else(|| "cosine".to_owned());
    DistanceMetric::parse(&value).map_err(|_| {
        usage("invalid --metric")
            .with_detail("field", "--metric")
            .with_detail("actual", &value)
            .with_detail("expected", "cosine|dot_product|euclidean")
    })
}

fn open_database_with_durability(path: &str, durability: Durability) -> Result<Database> {
    Database::open_with_options(
        path,
        DatabaseOpenOptions {
            durability: Some(durability),
            ..DatabaseOpenOptions::default()
        },
    )
}

fn help_json() -> JsonValue {
    object_from_pairs([
        ("schema_version", JsonValue::String("0.1".into())),
        ("format", JsonValue::String("json".into())),
        (
            "commands",
            JsonValue::Array(vec![
                help_command("help", "stable", "coffer help", &[]),
                help_command("init", "p0", "coffer init <path>", &[]),
                help_command(
                    "create-collection",
                    "p0",
                    "coffer create-collection <path> <name>",
                    &[
                        "--dim",
                        "--metric",
                        "--embedding-model",
                        "--tokenizer",
                        "--vectors-normalized",
                    ],
                ),
                help_command(
                    "list-collections",
                    "p0",
                    "coffer list-collections <path>",
                    &[],
                ),
                help_command(
                    "collection-stats",
                    "p0",
                    "coffer collection-stats <path>",
                    &["--collection", "--include-index"],
                ),
                help_command(
                    "import",
                    "p0",
                    "coffer import <path>",
                    &["--manifest", "--on-conflict", "--durability"],
                ),
                help_command(
                    "import-embeddings",
                    "p0",
                    "coffer import-embeddings <path> <input-path>",
                    &["--collection", "--on-conflict", "--durability"],
                ),
                help_command(
                    "search",
                    "p0",
                    &format!(
                        "coffer search <path> [<query>] [--top-k <1..={}>]",
                        crate::query::MAX_SEARCH_TOP_K
                    ),
                    &[
                        "--collection",
                        "--top-k",
                        "--profile",
                        "--filter",
                        "--weights",
                        "--metadata-boost",
                        "--embedding",
                    ],
                ),
                help_command(
                    "vector-search",
                    "p0",
                    &format!(
                        "coffer vector-search <path> [--top-k <1..={}>]",
                        crate::query::MAX_SEARCH_TOP_K
                    ),
                    &[
                        "--embedding",
                        "--collection",
                        "--top-k",
                        "--profile",
                        "--filter",
                    ],
                ),
                help_command(
                    "get-document",
                    "p0",
                    "coffer get-document <path>",
                    &["--collection", "--document-id", "--external-id"],
                ),
                help_command(
                    "get-chunk",
                    "p0",
                    "coffer get-chunk <path>",
                    &["--collection", "--chunk-id", "--chunk-external-id"],
                ),
                help_command(
                    "delete-document",
                    "p0",
                    "coffer delete-document <path>",
                    &["--collection", "--document-id", "--external-id"],
                ),
                help_command(
                    "delete-chunk",
                    "p0",
                    "coffer delete-chunk <path>",
                    &["--collection", "--chunk-id", "--chunk-external-id"],
                ),
                help_command(
                    "drop-collection",
                    "p0",
                    "coffer drop-collection <path>",
                    &["--collection"],
                ),
                help_command("stats", "p0", "coffer stats <path>", &["--include-index"]),
                help_command("check", "p0", "coffer check <path>", &["--verify-indexes"]),
                help_command(
                    "compact",
                    "p0",
                    "coffer compact <path>",
                    &["--rebuild-indexes"],
                ),
                help_command(
                    "benchmark",
                    "p0",
                    "coffer benchmark <path>",
                    &[
                        "--dataset",
                        "--profile",
                        "--durability",
                        "--collection",
                        "--output",
                        "--result-file",
                        "--require-baseline",
                    ],
                ),
                help_command(
                    "export",
                    "p0",
                    "coffer export <path> <output-dir>",
                    &["--collection", "--format"],
                ),
                help_command(
                    "why-not",
                    "debug",
                    &format!(
                        "coffer why-not <path> [--top-k <1..={}>]",
                        crate::query::MAX_SEARCH_TOP_K
                    ),
                    &[
                        "--collection",
                        "--query",
                        "--chunk-id",
                        "--chunk-external-id",
                        "--top-k",
                        "--profile",
                        "--filter",
                        "--weights",
                        "--embedding",
                        "--metadata-boost",
                    ],
                ),
                help_command(
                    "rebuild-index",
                    "debug",
                    "coffer rebuild-index <path> --collection <name>",
                    &["--collection"],
                ),
                help_command(
                    "repair",
                    "debug",
                    "coffer repair <path>",
                    &["--truncate-tail"],
                ),
                help_command(
                    "snapshot",
                    "future",
                    "coffer snapshot <path> <output-path>",
                    &[],
                ),
                help_command("profile-help", "stable", "coffer profile-help", &[]),
            ]),
        ),
        (
            "exit_codes",
            JsonValue::Array(vec![
                help_exit_code(0, "success"),
                help_exit_code(1, "user_or_input_error"),
                help_exit_code(2, "file_or_corruption_error"),
                help_exit_code(3, "check_issues"),
                help_exit_code(4, "lock_contention"),
            ]),
        ),
    ])
}

fn help_command(name: &str, stability: &str, synopsis: &str, flags: &[&str]) -> JsonValue {
    object_from_pairs([
        ("name", JsonValue::String(name.into())),
        ("stability", JsonValue::String(stability.into())),
        ("synopsis", JsonValue::String(synopsis.into())),
        (
            "flags",
            JsonValue::Array(
                flags
                    .iter()
                    .map(|flag| JsonValue::String((*flag).into()))
                    .collect(),
            ),
        ),
    ])
}

fn help_exit_code(code: u8, meaning: &str) -> JsonValue {
    object_from_pairs([
        ("code", JsonValue::Number(code as f64)),
        ("meaning", JsonValue::String(meaning.into())),
    ])
}

fn profile_help_json() -> JsonValue {
    object_from_pairs([
        ("schema_version", JsonValue::String("0.1".into())),
        (
            "default",
            JsonValue::String(ResourceProfile::Balanced.as_cli_str().into()),
        ),
        (
            "profiles",
            JsonValue::Array(vec![
                profile_help_item(
                    ResourceProfile::Expanded,
                    "advanced preset for retrieval experiments, candidate inspection, and quality evaluation, with larger default candidate windows and cache budget; may increase CPU, memory, or latency without improving Recall/MRR. Per-query selection preserves explicit runtime cache-budget overrides; cache budget is not a process RSS limit",
                ),
                profile_help_item(
                    ResourceProfile::Balanced,
                    "default preset for everyday local RAG and code search; choose balanced when unsure. Without metadata boost, text+vector defaults to BM25/vector weights 0.5/0.5; text-only or vector-only uses weight 1 for its input. With metadata boost, defaults are 0.45/0.50/0.05 for both inputs, 0.95/0/0.05 for text-only, and 0/0.95/0.05 for vector-only (BM25/vector/metadata). --weights overrides these defaults; explain records requested/effective weights and their source. External vectors do not guarantee semantic relevance; the engine remains responsible for correct calculation, filtering and configuration. Score shares are not recall shares or probabilities. Reference-recipe quality requires separate reviewed evaluation; user overrides do not validate the default path. Candidate and cache budgets control investment, not an optimality guarantee. Per-query selection preserves explicit runtime cache-budget overrides; cache budget is not a process RSS limit",
                ),
                profile_help_item(
                    ResourceProfile::Constrained,
                    "advanced preset for applications with limited host resources, with smaller default candidate windows and cache budget. Queries may be slower, retain fewer candidates, or report resource_exhausted when the budget is insufficient. This preset does not guarantee lower RSS, CPU use, or energy use. Lower candidate limits do not reduce full exact vector distance work; explicit runtime cache-budget overrides still apply, and cache budget is not a process RSS limit",
                ),
            ]),
        ),
    ])
}

fn profile_help_item(profile: ResourceProfile, description: &str) -> JsonValue {
    object_from_pairs([
        ("cli", JsonValue::String(profile.as_cli_str().into())),
        ("json", JsonValue::String(profile.as_json_str().into())),
        ("alias", JsonValue::Null),
        ("description", JsonValue::String(description.into())),
        (
            "candidate_limits",
            object_from_pairs([
                (
                    "bm25",
                    JsonValue::Number(profile.params().bm25_candidates as f64),
                ),
                (
                    "vector",
                    JsonValue::Number(profile.params().vector_candidates as f64),
                ),
                (
                    "fusion",
                    JsonValue::Number(profile.params().fusion_candidates as f64),
                ),
            ]),
        ),
        (
            "cache_budget_mb",
            JsonValue::Number(profile.params().cache_budget_mb as f64),
        ),
    ])
}

fn search_command(mut args: Args, vector_only: bool) -> Result<CommandResult> {
    let path = args.required("path")?;
    let context = JsonOperationContext::standalone();
    let mut input_work = context.reservation();
    let mut filter_work = context.reservation();
    if vector_only && args.args.iter().any(|arg| arg == "--weights") {
        return Err(usage("vector-search does not accept --weights")
            .with_detail("command", "vector-search")
            .with_detail("field", "--weights")
            .with_detail("reason", "unsupported_option"));
    }
    if vector_only && args.args.iter().any(|arg| arg == "--metadata-boost") {
        return Err(usage("vector-search does not accept --metadata-boost")
            .with_detail("command", "vector-search")
            .with_detail("field", "--metadata-boost")
            .with_detail("reason", "unsupported_option"));
    }
    let embedding = args
        .flag_value("--embedding")?
        .map(|path| -> Result<_> {
            let (value, work) = read_embedding_file_in(&path, &context)?;
            input_work.merge(work);
            Ok(value)
        })
        .transpose()?;
    if vector_only && embedding.is_none() {
        return Err(usage("vector-search requires --embedding"));
    }
    let collection = args
        .flag_value("--collection")?
        .ok_or_else(|| usage("--collection is required"))?;
    let top_k = parse_top_k_flag(&mut args)?;
    let profile = parse_profile_flag(&mut args)?;
    if matches!(profile, Some(ParsedProfile::Help)) {
        args.reject_remaining()?;
        return Ok(CommandResult::plain(profile_help_json()));
    }
    let QueryJsonOptions {
        filter,
        weights,
        metadata_boost,
    } = parse_query_options_in(&mut args, &context, &mut input_work, &mut filter_work)?;
    let query = if vector_only { None } else { args.next() };
    if !vector_only
        && query
            .as_ref()
            .map(|query| query.trim().is_empty())
            .unwrap_or(true)
        && embedding.is_none()
    {
        return Err(usage("search requires query text or --embedding"));
    }
    args.reject_remaining()?;
    let db = Database::open_read_only(path)?;
    let mut builder = db
        .collection(&collection)?
        .search()
        .top_k(top_k)
        .filter(filter);
    if let Some(metadata_boost) = metadata_boost {
        builder = builder.metadata_boost(metadata_boost);
    }
    if let Some(ParsedProfile::Profile(profile)) = profile {
        builder = builder.resource_profile(profile);
    }
    if let Some(weights) = weights {
        builder = builder.weights(weights);
    }
    if let Some(query) = query {
        builder = builder.text(query);
    }
    if let Some(embedding) = embedding {
        builder = builder.query_embedding(embedding);
    }
    let (result, mut work) = builder.execute_retained(Some(context.clone()))?;
    #[cfg(test)]
    tests::before_query_dto(tests::QueryDtoStage::InputRelease, &context, &work);
    drop(input_work);
    drop(filter_work);
    #[cfg(test)]
    tests::before_query_dto(tests::QueryDtoStage::DatabaseClose, &context, &work);
    drop(db);
    Ok(CommandResult::retained(
        result.into_json_retained(&mut work)?,
        context,
        work.into_json(),
        JsonOutputSchema::SearchEnvelope,
    ))
}

fn why_not_command(mut args: Args) -> Result<CommandResult> {
    let path = args.required("path")?;
    let context = JsonOperationContext::standalone();
    let mut input_work = context.reservation();
    let mut filter_work = context.reservation();
    let collection = args
        .flag_value("--collection")?
        .ok_or_else(|| usage("--collection is required"))?;
    let expected_chunk_id = match args.flag_value("--chunk-id")? {
        Some(id) => id,
        None => args
            .flag_value("--chunk-external-id")?
            .ok_or_else(|| usage("--chunk-id or --chunk-external-id is required"))?,
    };
    let query = args.flag_value("--query")?;
    let top_k = parse_top_k_flag(&mut args)?;
    let profile = parse_profile_flag(&mut args)?;
    if matches!(profile, Some(ParsedProfile::Help)) {
        args.reject_remaining()?;
        return Ok(CommandResult::plain(profile_help_json()));
    }
    let embedding = args
        .flag_value("--embedding")?
        .map(|path| -> Result<_> {
            let (value, work) = read_embedding_file_in(&path, &context)?;
            input_work.merge(work);
            Ok(value)
        })
        .transpose()?;
    let QueryJsonOptions {
        filter,
        weights,
        metadata_boost,
    } = parse_query_options_in(&mut args, &context, &mut input_work, &mut filter_work)?;
    args.reject_remaining()?;
    let db = Database::open_read_only(path)?;
    let (result, mut trace_work) = db.why_not_debug_with_metadata_boost_in(
        &collection,
        &expected_chunk_id,
        query,
        embedding,
        filter,
        weights,
        metadata_boost.is_some(),
        metadata_boost,
        top_k,
        profile.map(|profile| match profile {
            ParsedProfile::Profile(profile) => profile,
            ParsedProfile::Help => unreachable!("handled above"),
        }),
        Some(context.clone()),
    )?;
    trace_work.merge(filter_work);
    #[cfg(test)]
    tests::before_query_dto(tests::QueryDtoStage::DatabaseClose, &context, &trace_work);
    drop(db);
    #[cfg(test)]
    tests::before_query_dto(tests::QueryDtoStage::InputRelease, &context, &trace_work);
    drop(input_work);
    let result = result.into_json_retained(&mut trace_work)?;
    Ok(CommandResult::retained(
        result,
        context,
        trace_work.into_json(),
        JsonOutputSchema::WhyNotEnvelope,
    ))
}

fn benchmark_command(mut args: Args) -> Result<CommandResult> {
    let path = args.required("path")?;
    let profile = parse_profile_flag(&mut args)?;
    if matches!(profile, Some(ParsedProfile::Help)) {
        args.reject_remaining()?;
        return Ok(CommandResult::plain(profile_help_json()));
    }
    let dataset_path = args
        .flag_value("--dataset")?
        .ok_or_else(|| usage("--dataset is required"))?;
    let durability = parse_durability_flag(&mut args)?;
    let output_path = args.flag_value("--output")?;
    let result_file = args.flag_value("--result-file")?;
    let require_baseline = args.has_flag("--require-baseline");
    let result_file = match (output_path, result_file) {
        (Some(_), Some(_)) => return Err(usage("use only one of --output or --result-file")),
        (Some(path), None) | (None, Some(path)) => Some(path),
        (None, None) => None,
    };
    let collection = args
        .flag_value("--collection")?
        .unwrap_or_else(|| "benchmark".into());
    args.reject_remaining()?;
    let profile = profile
        .map(|profile| match profile {
            ParsedProfile::Profile(profile) => profile,
            ParsedProfile::Help => unreachable!("handled above"),
        })
        .unwrap_or(ResourceProfile::Balanced);
    let output = crate::benchmark::run_benchmark(crate::benchmark::BenchmarkOptions {
        path,
        dataset_path,
        collection,
        profile,
        durability,
        result_file,
        require_baseline,
    })?;
    Ok(CommandResult::retained(
        output.value,
        output.context,
        output.work,
        JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
    ))
}

fn with_json_argument<T>(
    value: String,
    kind: JsonInputKind,
    context: &JsonOperationContext,
    parse: impl FnOnce(&str) -> Result<T>,
) -> Result<T> {
    let mut input = context.reservation();
    input.grow(value.capacity(), kind)?;
    let result = parse(&value);
    drop(value);
    result
}

struct QueryJsonOptions {
    filter: JsonValue,
    weights: Option<SearchWeights>,
    metadata_boost: Option<MetadataBoostConfig>,
}

fn parse_query_options_in(
    args: &mut Args,
    context: &JsonOperationContext,
    input_work: &mut JsonReservation,
    filter_work: &mut JsonReservation,
) -> Result<QueryJsonOptions> {
    let filter = args
        .flag_value("--filter")?
        .map(|value| -> Result<_> {
            let (value, work) =
                with_json_argument(value, JsonInputKind::Filter, context, |value| {
                    crate::json::parse_with_schema_in(
                        value,
                        JsonSchema::Value(JsonInputKind::Filter),
                        context,
                    )
                })?
                .into_parts();
            filter_work.merge(work);
            Ok(value)
        })
        .transpose()?
        .unwrap_or_else(JsonValue::object);
    let weights = args
        .flag_value("--weights")?
        .map(|value| {
            with_json_argument(value, JsonInputKind::Weights, context, |value| {
                parse_weights_in(value, context)
            })
        })
        .transpose()?;
    let metadata_boost = args
        .flag_value("--metadata-boost")?
        .map(|value| -> Result<_> {
            let (value, work) = if value.trim_start().starts_with('{') {
                with_json_argument(value, JsonInputKind::MetadataBoost, context, |value| {
                    parse_metadata_boost_in(value, context)
                })?
            } else {
                parse_metadata_boost_in(&value, context)?
            };
            input_work.merge(work);
            Ok(value)
        })
        .transpose()?;
    Ok(QueryJsonOptions {
        filter,
        weights,
        metadata_boost,
    })
}

#[cfg(test)]
fn parse_weights(value: &str) -> Result<SearchWeights> {
    parse_weights_in(value, &JsonOperationContext::standalone())
}

fn parse_weights_in(value: &str, context: &JsonOperationContext) -> Result<SearchWeights> {
    let value = crate::json::parse_with_schema_in(
        value,
        JsonSchema::Value(JsonInputKind::Weights),
        context,
    )
    .map_err(|err| {
        if crate::json::is_limit_error(&err)
            || matches!(err.code(), ErrorCode::IoError | ErrorCode::PermissionDenied)
        {
            return err.with_detail("field", "weights");
        }
        Error::new(
            ErrorCode::InvalidSearchWeights,
            format!("invalid search weights JSON: {}", err.message()),
        )
        .with_detail("field", "weights")
        .with_detail("reason", "invalid_json")
        .with_detail("weights_reason", "invalid_json")
    })?;
    let obj = value.value.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidSearchWeights,
            "--weights must be a JSON object",
        )
        .with_detail("field", "weights")
        .with_detail("weights_reason", "invalid_type")
        .with_detail("expected", "object")
    })?;
    for key in obj.keys() {
        if !matches!(key.as_str(), "bm25" | "vector" | "metadata") {
            return Err(Error::new(
                ErrorCode::InvalidSearchWeights,
                format!("--weights contains unsupported field {key}"),
            )
            .with_detail("field", format!("weights.{key}"))
            .with_detail("weights_reason", "unsupported_field")
            .with_detail("reason", "unsupported_field"));
        }
    }
    SearchWeights::new(
        required_weight(obj, "bm25")?,
        required_weight(obj, "vector")?,
        required_weight(obj, "metadata")?,
    )
}

#[cfg(test)]
fn parse_metadata_boost(value: &str) -> Result<MetadataBoostConfig> {
    parse_metadata_boost_in(value, &JsonOperationContext::standalone()).map(|(value, _)| value)
}

fn parse_metadata_boost_in(
    value: &str,
    context: &JsonOperationContext,
) -> Result<(MetadataBoostConfig, JsonReservation)> {
    let parsed = if value.trim_start().starts_with('{') {
        crate::json::parse_with_schema_in(
            value,
            JsonSchema::Value(JsonInputKind::MetadataBoost),
            context,
        )
    } else {
        crate::json::parse_reader_in(
            fs::File::open(value)?,
            JsonSchema::Value(JsonInputKind::MetadataBoost),
            context,
        )
    };
    let parsed = parsed.map_err(|err| {
        if crate::json::is_limit_error(&err)
            || matches!(err.code(), ErrorCode::IoError | ErrorCode::PermissionDenied)
        {
            return err.with_detail("field", "metadata_boost");
        }
        Error::new(
            ErrorCode::InvalidMetadataBoost,
            format!("invalid metadata boost JSON: {}", err.message()),
        )
        .with_detail("field", "metadata_boost")
        .with_detail("metadata_boost_reason", "invalid_json")
        .with_detail("reason", "invalid_json")
    })?;
    MetadataBoostConfig::parse_validated_in(&parsed.value, context)
}

fn required_weight(obj: &crate::json::JsonObject, field: &str) -> Result<f64> {
    obj.get(field).and_then(JsonValue::as_f64).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidSearchWeights,
            format!("--weights must include numeric {field}"),
        )
        .with_detail("field", format!("weights.{field}"))
        .with_detail(
            "weights_reason",
            if obj.contains_key(field) {
                "invalid_type"
            } else {
                "missing_required_field"
            },
        )
        .with_detail("expected", "number")
    })
}

#[cfg(test)]
fn read_embedding_file(path: &str) -> Result<QueryEmbeddingInput> {
    read_embedding_file_in(path, &JsonOperationContext::standalone()).map(|(value, _)| value)
}

fn read_embedding_file_in(
    path: &str,
    context: &JsonOperationContext,
) -> Result<(QueryEmbeddingInput, JsonReservation)> {
    let value = crate::json::parse_reader_in(
        fs::File::open(path)?,
        JsonSchema::QueryEmbeddingObject,
        context,
    )
    .map_err(|error| error.with_detail("path", path))?;
    let obj = value.value.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            "query_embedding.json must be a JSON object",
        )
        .with_detail("path", path)
    })?;
    for field in obj.keys() {
        if !matches!(field.as_str(), "model_name" | "dim" | "vector") {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "query_embedding.json contains unsupported field",
            )
            .with_detail("field", field)
            .with_detail("path", path));
        }
    }
    let model_name = obj
        .get("model_name")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "query_embedding.json must include string model_name",
            )
            .with_detail("field", "model_name")
            .with_detail("path", path)
        })?;
    if model_name.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::MissingRequiredField,
            "query_embedding.json model_name must not be empty",
        )
        .with_detail("field", "model_name")
        .with_detail("path", path));
    }
    let dim = obj.get("dim").and_then(JsonValue::as_f64).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            "query_embedding.json must include numeric dim",
        )
        .with_detail("field", "dim")
        .with_detail("path", path)
    })?;
    if !dim.is_finite() || dim <= 0.0 || dim.fract() != 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidVectorLength,
            "query_embedding.json dim must be a positive integer",
        )
        .with_detail("field", "dim")
        .with_detail("path", path));
    }
    let dim = dim as usize;
    let array = obj
        .get("vector")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "query_embedding.json must include vector array",
            )
            .with_detail("field", "vector")
            .with_detail("path", path)
        })?;
    let mut work = context.reservation();
    let mut vector = Vec::new();
    work.reserve_vec(&mut vector, array.len(), JsonInputKind::QueryEmbedding)?;
    let values = array.iter().map(|value| {
        let value = value.as_f64().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidVectorLength,
                "query_embedding.json vector entries must be numbers",
            )
            .with_detail("field", "vector")
            .with_detail("path", path)
        })?;
        let value = value as f32;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "query_embedding.json vector entries must be finite f32 values",
            )
            .with_detail("field", "vector")
            .with_detail("path", path))
        }
    });
    for value in values {
        vector.push(value?);
    }
    if vector.len() != dim {
        return Err(Error::new(
            ErrorCode::InvalidVectorLength,
            "query_embedding.json dim must match vector length",
        )
        .with_detail("field", "dim")
        .with_detail("dim", dim)
        .with_detail("actual_vector_length", vector.len())
        .with_detail("path", path));
    }
    Ok((
        QueryEmbeddingInput {
            model_name: Some(work.copy_string(model_name, JsonInputKind::JsonlRecord)?),
            dim,
            vector,
        },
        work,
    ))
}

#[derive(Clone)]
struct Args {
    args: Vec<String>,
    pos: usize,
}

impl Args {
    fn new(args: Vec<String>) -> Self {
        Self {
            args: args.into_iter().filter(|arg| arg != "--json").collect(),
            pos: 0,
        }
    }

    fn next(&mut self) -> Option<String> {
        let value = self.args.get(self.pos).cloned();
        if value.is_some() {
            self.pos += 1;
        }
        value
    }

    fn required(&mut self, name: &str) -> Result<String> {
        self.next().ok_or_else(|| {
            usage(format!("{name} is required"))
                .with_detail("field", name)
                .with_detail("reason", "missing_required_argument")
        })
    }

    fn flag_value(&mut self, flag: &str) -> Result<Option<String>> {
        if let Some(index) = self.args.iter().position(|arg| arg == flag) {
            if index + 1 >= self.args.len() {
                return Err(usage(format!("{flag} requires a value"))
                    .with_detail("field", flag)
                    .with_detail("reason", "missing_value"));
            }
            let value = self.args.remove(index + 1);
            self.args.remove(index);
            if self.pos > index {
                self.pos = self.pos.saturating_sub(1);
            }
            if self.pos > index {
                self.pos = self.pos.saturating_sub(1);
            }
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    fn has_flag(&mut self, flag: &str) -> bool {
        if let Some(index) = self.args.iter().position(|arg| arg == flag) {
            self.args.remove(index);
            true
        } else {
            false
        }
    }

    fn reject_remaining(&self) -> Result<()> {
        if let Some(arg) = self.args.get(self.pos) {
            return Err(usage(format!("unexpected argument {arg}"))
                .with_detail("field", "argument")
                .with_detail("actual", arg)
                .with_detail("reason", "unexpected_argument"));
        }
        Ok(())
    }
}

fn usage(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::CliUsage, message)
}

#[cfg(test)]
mod tests {
    mod catalog_output;
    mod diagnostic_output;
    mod embedding_import;
    mod import_output;
    mod query_dto;
    mod summary_output;
    use super::*;
    use crate::model::{ChunkInput, DocumentInput, EmbeddingInput};

    #[derive(Clone, Copy, Eq, PartialEq)]
    pub(super) enum QueryDtoStage {
        InputRelease,
        DatabaseClose,
    }

    type QueryDtoHook = Box<dyn FnOnce(&JsonOperationContext, &crate::storage::RecordReservation)>;
    thread_local! {
        static QUERY_DTO_READY: std::cell::RefCell<Option<(QueryDtoStage, QueryDtoHook)>> = const { std::cell::RefCell::new(None) };
    }

    pub(super) fn before_query_dto(
        stage: QueryDtoStage,
        context: &JsonOperationContext,
        work: &crate::storage::RecordReservation,
    ) {
        let hook = QUERY_DTO_READY.with(|slot| {
            let mut pending = slot.borrow_mut();
            if pending
                .as_ref()
                .is_some_and(|(expected, _)| *expected == stage)
            {
                pending.take().map(|(_, hook)| hook)
            } else {
                None
            }
        });
        if let Some(hook) = hook {
            hook(context, work);
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("cofferdb-cli-{name}-{}.coffer", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn owned_json_arguments_reserve_capacity_before_parsing() {
        let context = JsonOperationContext::standalone();
        let mut held = context.reservation();
        held.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 512,
            JsonInputKind::Filter,
        )
        .unwrap();
        let mut raw = String::with_capacity(1024);
        raw.push_str("{}");
        let invoked = std::cell::Cell::new(false);
        let error = with_json_argument(raw, JsonInputKind::Filter, &context, |_| {
            invoked.set(true);
            Ok(())
        })
        .unwrap_err();
        assert!(!invoked.get());
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        drop(held);
        with_json_argument("{}".into(), JsonInputKind::Filter, &context, |_| Ok(())).unwrap();
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::Filter,
            )
            .unwrap();
    }

    #[test]
    fn search_preprocessing_shares_inputs_and_core_does_not_charge_filter_twice() {
        let path = temp_path("query-json-handoff");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        drop(db);
        let filter = format!(
            "{{\"chunk.x\":{{\"$in\":[{}null]}}}}",
            "null,".repeat(450_000)
        );
        let args = |filter: String| {
            vec![
                "coffer".into(),
                "search".into(),
                path.display().to_string(),
                "missing".into(),
                "--collection".into(),
                "notes".into(),
                "--filter".into(),
                filter,
            ]
        };
        let output = run(args(filter.clone())).unwrap();
        assert!(parse(&output)
            .unwrap()
            .get("result")
            .unwrap()
            .get("results")
            .unwrap()
            .as_array()
            .unwrap()
            .is_empty());
        let boost = format!(
            r#"{{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{{"when":{{"chunk.y":{{"$in":[{}null]}}}},"boost":0.1}}]}}"#,
            "null,".repeat(350_000)
        );
        parse_metadata_boost(&boost).unwrap();
        let why_not = vec![
            "coffer".into(),
            "why-not".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            "missing".into(),
            "--query".into(),
            "missing".into(),
            "--filter".into(),
            filter.clone(),
            "--metadata-boost".into(),
            boost.clone(),
        ];
        let error = run(why_not).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataBoost);
        assert_eq!(
            detail_value(&error, "json_input_kind"),
            Some("metadata_boost")
        );
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        let mut combined = args(filter);
        combined.extend(["--metadata-boost".into(), boost]);
        let error = run(combined).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataBoost);
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        assert_eq!(
            detail_value(&error, "json_input_kind"),
            Some("metadata_boost")
        );
        std::fs::remove_file(path).unwrap();
    }

    fn create_cli_schema_fixture(name: &str) -> (PathBuf, String, String) {
        create_cli_schema_fixture_with_model(name, "m")
    }

    fn create_cli_schema_fixture_with_model(name: &str, model: &str) -> (PathBuf, String, String) {
        let path = temp_path(name);
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: Some("Doc".into()),
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "stable cli result schema".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: model.into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        let document_id = col.get_document_by_external_id("doc").unwrap().document_id;
        let chunk_id = col.get_chunk_by_external_id("doc#1").unwrap().chunk_id;
        drop(col);
        drop(db);
        (path, document_id, chunk_id)
    }

    #[test]
    fn retained_benchmark_envelopes_keep_warnings_and_release_budget() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let warning = object_from_pairs([
                (
                    "code",
                    JsonValue::String("benchmark_baseline_not_run".into()),
                ),
                ("message", JsonValue::String("baseline unavailable".into())),
                ("reason", JsonValue::String("x".repeat(seed))),
            ]);
            let result = object_from_pairs([
                ("warnings", JsonValue::Array(vec![warning])),
                (
                    "status",
                    JsonValue::String("completed_cofferdb_runs".into()),
                ),
            ]);
            let expected = cli_envelope("benchmark", true, result.clone(), None, 7);
            let work = result
                .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &context)
                .unwrap();
            let actual = success_output(
                "benchmark",
                CommandResult::retained(
                    result,
                    context.clone(),
                    work,
                    JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
                ),
                7,
            )
            .unwrap();
            assert_eq!(actual, expected);
            let parsed = parse(&actual).unwrap();
            let warning = &parsed.get("warnings").unwrap().as_array().unwrap()[0];
            assert_eq!(warning.get("severity").unwrap().as_str(), Some("warning"));
            assert_eq!(
                warning
                    .get("details")
                    .unwrap()
                    .get("reason")
                    .unwrap()
                    .as_str(),
                Some("x".repeat(seed).as_str())
            );
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        let context = JsonOperationContext::standalone();
        let result = JsonValue::String("\"".repeat(20_000));
        let work = result
            .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &context)
            .unwrap();
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes() - 4000,
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
        let error = success_output(
            "benchmark",
            CommandResult::retained(
                result,
                context.clone(),
                work,
                JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
            ),
            7,
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(
            detail_value(&error, "json_input_kind"),
            Some("benchmark_artifact")
        );
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        drop(busy);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
    }

    #[test]
    fn query_envelopes_reserve_construction_and_preserve_the_wire_format() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let result = object_from_pairs([("query", JsonValue::String(format!("query-{seed}")))]);
            let expected = cli_envelope("search", true, result.clone(), None, seed);
            let mut work = result
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            let envelope = build_cli_envelope(
                "search",
                true,
                result,
                JsonValue::Array(Vec::new()),
                None,
                seed,
                &JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson),
            )
            .unwrap();
            let measured = envelope
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            drop(measured);
            let (actual, _) = envelope
                .pretty_retained_in(JsonOutputSchema::SearchEnvelope, &context, &mut work)
                .unwrap();
            assert_eq!(actual, expected);
        }

        let context = JsonOperationContext::standalone();
        let mut occupied = context.reservation();
        occupied
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 64,
                JsonInputKind::StoredJson,
            )
            .unwrap();
        let mut work = context.reservation();
        let error = build_cli_envelope(
            "search",
            true,
            JsonValue::Null,
            JsonValue::Array(Vec::new()),
            None,
            0,
            &JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson),
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(
            error
                .detail()
                .iter()
                .find(|(key, _)| key == "json_input_kind")
                .map(|(_, value)| value.as_str()),
            Some("stored_json")
        );
        drop(work);
        drop(occupied);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::StoredJson,
            )
            .unwrap();
    }

    #[test]
    fn cli_search_keeps_result_json_live_while_budgeting_serialized_output() {
        let (path, _, _) = create_cli_schema_fixture("search-output-budget");
        let db = Database::open(&path).unwrap();
        let collection = db.collection("notes").unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "wide".into(),
                title: None,
                source_uri: None,
                metadata: object_from_pairs([(
                    "payload",
                    JsonValue::String("x".repeat(4 * 1024 * 1024)),
                )]),
                chunks: (0..5)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("wide#{ordinal}"),
                        text: "payload".into(),
                        ordinal,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        assert_eq!(
            collection
                .search()
                .text("payload")
                .top_k(5)
                .execute()
                .unwrap()
                .results
                .len(),
            5
        );
        drop(collection);
        drop(db);
        let before = fs::read(&path).unwrap();
        let args = |top_k: &str| {
            vec![
                "coffer".into(),
                "search".into(),
                path.display().to_string(),
                "payload".into(),
                "--collection".into(),
                "notes".into(),
                "--top-k".into(),
                top_k.into(),
            ]
        };
        let (output, code) = run_with_exit_code(args("2")).unwrap();
        assert_eq!(code, 0);
        assert!(output.len() > crate::json::JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES);
        drop(output);
        let error = run(args("5")).unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        assert_eq!(detail_value(&error, "json_input_kind"), Some("stored_json"));
        assert_eq!(fs::read(&path).unwrap(), before);
        run(args("1")).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn init_output_uses_success_envelope() {
        let path = temp_path("init-envelope");
        let output = run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("init")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        let path_str = path.display().to_string();
        assert_eq!(
            result.get("path").and_then(JsonValue::as_str),
            Some(path_str.as_str())
        );
        assert_eq!(
            result.get("created").and_then(JsonValue::as_bool),
            Some(true)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rebuild_index_output_uses_success_envelope() {
        let path = temp_path("rebuild-index-envelope");
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

        let output = run(vec![
            "coffer".into(),
            "rebuild-index".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("rebuild-index")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("collection").and_then(JsonValue::as_str),
            Some("notes")
        );
        assert_eq!(
            result
                .get("index_suffix_status")
                .and_then(JsonValue::as_str),
            Some("persistent_checkpoint")
        );
        assert_eq!(
            result.get("rebuild_needed").and_then(JsonValue::as_bool),
            Some(false)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn create_collection_accepts_vectors_normalized_flag() {
        let path = temp_path("create-vectors-normalized");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
            "--metric".into(),
            "dot_product".into(),
            "--vectors-normalized".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result
                .get("vectors_normalized")
                .and_then(JsonValue::as_bool),
            Some(true)
        );

        let listed = run(vec![
            "coffer".into(),
            "list-collections".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let json = parse(&listed).unwrap();
        let collections = json
            .get("result")
            .and_then(|result| result.get("collections"))
            .and_then(JsonValue::as_array)
            .unwrap();
        assert_eq!(
            collections[0]
                .get("vectors_normalized")
                .and_then(JsonValue::as_bool),
            Some(true)
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn create_collection_rejects_unknown_arguments() {
        let path = temp_path("create-rejects-unknown");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();

        let err = run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
            "--unknown".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("unexpected argument --unknown"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn invalid_create_collection_dim_reports_machine_readable_detail() {
        let text_err = run(vec![
            "coffer".into(),
            "create-collection".into(),
            "notes.coffer".into(),
            "notes".into(),
            "--dim".into(),
            "wide".into(),
        ])
        .unwrap_err();
        assert_eq!(text_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&text_err, "field"), Some("--dim"));
        assert_eq!(detail_value(&text_err, "actual"), Some("wide"));
        assert_eq!(
            detail_value(&text_err, "expected"),
            Some("positive_integer")
        );

        let zero_err = run(vec![
            "coffer".into(),
            "create-collection".into(),
            "notes.coffer".into(),
            "notes".into(),
            "--dim".into(),
            "0".into(),
        ])
        .unwrap_err();
        assert_eq!(zero_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&zero_err, "field"), Some("--dim"));
        assert_eq!(detail_value(&zero_err, "actual"), Some("0"));
        assert_eq!(
            detail_value(&zero_err, "expected"),
            Some("positive_integer")
        );
    }

    #[test]
    fn invalid_create_collection_metric_reports_machine_readable_detail() {
        let err = run(vec![
            "coffer".into(),
            "create-collection".into(),
            "notes.coffer".into(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
            "--metric".into(),
            "angular".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&err, "field"), Some("--metric"));
        assert_eq!(detail_value(&err, "actual"), Some("angular"));
        assert_eq!(
            detail_value(&err, "expected"),
            Some("cosine|dot_product|euclidean")
        );
    }

    #[test]
    fn query_option_parsers_preserve_native_and_read_errors_without_user_error_reasons() {
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let weights = r#"{"bm25":1,"vector":0,"metadata":0}"#;
        let error =
            crate::test_alloc::fail_nth_allocation_once(1, || parse_weights_in(weights, &context))
                .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(detail_value(&error, "field"), Some("weights"));
        assert_eq!(detail_value(&error, "weights_reason"), None);
        assert_eq!(detail_value(&error, "json_limit_reason"), None);
        assert_eq!(context.available_workspace_bytes(), capacity);
        parse_weights_in(weights, &context).unwrap();
        let boost = r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"chunk.x":true},"boost":0.1}]}"#;
        let error = crate::test_alloc::fail_nth_allocation_once(1, || {
            parse_metadata_boost_in(boost, &context)
        })
        .err()
        .unwrap();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(detail_value(&error, "field"), Some("metadata_boost"));
        assert_eq!(detail_value(&error, "metadata_boost_reason"), None);
        assert_eq!(detail_value(&error, "json_limit_reason"), None);
        assert_eq!(context.available_workspace_bytes(), capacity);
        drop(parse_metadata_boost_in(boost, &context).unwrap());
        let directory = temp_path("boost-read-error");
        fs::create_dir(&directory).unwrap();
        let error = parse_metadata_boost_in(directory.to_str().unwrap(), &context)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(detail_value(&error, "metadata_boost_reason"), None);
        assert_eq!(detail_value(&error, "json_limit_reason"), None);
        let envelope = parse(&error_output(Some("search"), &error)).unwrap();
        assert_eq!(
            envelope.get("error").unwrap().get("code").unwrap().as_str(),
            Some("io_error")
        );
        assert_eq!(context.available_workspace_bytes(), capacity);
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn metadata_boost_parser_accepts_file_path() {
        let path = temp_path("metadata-boost-config-json");
        std::fs::write(
            &path,
            r#"{
                "schema_version":"0.1",
                "rules":[{"when":{"chunk.importance":{"$gte":0.8}},"boost":0.1}],
                "combine":"sum",
                "max_boost":0.2
            }"#,
        )
        .unwrap();

        let config = parse_metadata_boost(&path.display().to_string()).unwrap();
        assert_eq!(config.rule_count(), 1);
        assert_eq!(config.hash().len(), 64);

        let inline = parse_metadata_boost(
            r#"{"schema_version":"0.1","rules":[{"when":{"chunk.importance":{"$gte":0.8}},"boost":0.1}],"combine":"sum","max_boost":0.2}"#,
        )
        .unwrap();
        assert_eq!(inline.rule_count(), 1);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn metadata_boost_parser_reports_invalid_metadata_boost_for_bad_json() {
        let err = parse_metadata_boost("{bad json").unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidMetadataBoost);
        assert!(err.message().contains("invalid metadata boost JSON"));
        assert_eq!(detail_value(&err, "field"), Some("metadata_boost"));
        assert_eq!(detail_value(&err, "reason"), Some("invalid_json"));
    }

    #[test]
    fn search_reports_invalid_metadata_boost_for_bad_json() {
        let err = run(vec![
            "coffer".into(),
            "search".into(),
            "notes.coffer".into(),
            "wal".into(),
            "--collection".into(),
            "notes".into(),
            "--metadata-boost".into(),
            "{bad json".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidMetadataBoost);
        assert!(err.message().contains("invalid metadata boost JSON"));
    }

    #[test]
    fn repair_truncate_tail_removes_corrupt_suffix_only() {
        use std::io::Write;

        let path = temp_path("repair-truncate-tail");
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
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"broken-index-suffix")
            .unwrap();

        let before = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(before.index_suffix_status, "rebuild_needed");
        assert!(before.repair_available);
        assert!(before.issues.iter().any(|issue| {
            issue
                .suggested_command
                .as_deref()
                .map(|command| {
                    command.contains("coffer rebuild-index")
                        && command.contains("--collection notes")
                })
                .unwrap_or(false)
        }));

        let output = run(vec![
            "coffer".into(),
            "repair".into(),
            path.display().to_string(),
            "--truncate-tail".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("repair")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("action").and_then(JsonValue::as_str),
            Some("truncate_tail")
        );
        assert_eq!(
            result.get("repaired").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert!(result
            .get("bytes_removed")
            .and_then(JsonValue::as_f64)
            .map(|bytes| bytes > 0.0)
            .unwrap_or(false));

        let after = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(after.status, "ok");
        assert_eq!(after.index_suffix_status, "missing");
        assert_eq!(after.live_documents, 0);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            after.critical_end_offset
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_command_uses_check_issue_exit_code() {
        use std::io::Write;

        let path = temp_path("check-issue-exit-code");
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
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"corrupt-tail")
            .unwrap();

        let (output, exit_code) = run_with_exit_code(vec![
            "coffer".into(),
            "check".into(),
            path.display().to_string(),
        ])
        .unwrap();
        assert_eq!(exit_code, 3);
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("check")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("status").and_then(JsonValue::as_str),
            Some("warning")
        );
        assert_eq!(
            result.get("repair_available").and_then(JsonValue::as_bool),
            Some(true)
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_accepts_advertised_verify_indexes_flag() {
        let path = temp_path("check-verify-indexes-flag");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "check".into(),
            path.display().to_string(),
            "--verify-indexes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("check")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(result.get("status").and_then(JsonValue::as_str), Some("ok"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn snapshot_command_copies_consistent_database() {
        let path = temp_path("snapshot-source");
        let output_path = temp_path("snapshot-output");
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
        let source_len = std::fs::metadata(&path).unwrap().len();

        let output = run(vec![
            "coffer".into(),
            "snapshot".into(),
            path.display().to_string(),
            output_path.display().to_string(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("snapshot")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("bytes_copied").and_then(JsonValue::as_f64),
            Some(source_len as f64)
        );
        assert_eq!(
            result
                .get("index_suffix_status")
                .and_then(JsonValue::as_str),
            Some("persistent_checkpoint")
        );
        assert_eq!(std::fs::metadata(&output_path).unwrap().len(), source_len);

        let snapshot_collections = run(vec![
            "coffer".into(),
            "list-collections".into(),
            output_path.display().to_string(),
        ])
        .unwrap();
        let snapshot_json = parse(&snapshot_collections).unwrap();
        let collections = snapshot_json
            .get("result")
            .and_then(JsonValue::as_object)
            .unwrap()
            .get("collections")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert_eq!(
            collections[0].get("name").and_then(JsonValue::as_str),
            Some("notes")
        );

        let err = run(vec![
            "coffer".into(),
            "snapshot".into(),
            path.display().to_string(),
            output_path.display().to_string(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchParams);

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn snapshot_command_requires_writer_lock() {
        let path = temp_path("snapshot-lock-source");
        let output_path = temp_path("snapshot-lock-output");
        let _db = Database::create(&path, DatabaseConfig::default()).unwrap();

        let err = run(vec![
            "coffer".into(),
            "snapshot".into(),
            path.display().to_string(),
            output_path.display().to_string(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::LockContention);

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn get_commands_use_public_result_schema() {
        let (path, _document_id, chunk_id) = create_cli_schema_fixture("get-schema");
        let output = run(vec![
            "coffer".into(),
            "get-document".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--external-id".into(),
            "doc".into(),
            "--include-chunks".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(result.get("document"), Some(JsonValue::Object(_))));
        assert_eq!(
            result.get("external_id").and_then(JsonValue::as_str),
            Some("doc")
        );
        assert_eq!(
            result.get("chunks_count").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result
                .get("chunks")
                .and_then(JsonValue::as_array)
                .map(|chunks| chunks.len()),
            Some(1)
        );

        let output = run(vec![
            "coffer".into(),
            "get-chunk".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            chunk_id,
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(result.get("chunk"), Some(JsonValue::Object(_))));
        assert!(matches!(result.get("document"), Some(JsonValue::Object(_))));
        assert_eq!(
            result.get("chunk_external_id").and_then(JsonValue::as_str),
            Some("doc#1")
        );
        assert_eq!(
            result.get("embedding_present").and_then(JsonValue::as_bool),
            Some(true)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn get_commands_serialize_full_depth_metadata_without_wrapper_penalty() {
        let (path, _, _) = create_cli_schema_fixture("get-deep-metadata");
        let db = Database::open(&path).unwrap();
        let col = db.collection("notes").unwrap();
        let mut nested = JsonValue::String("depth-boundary".into());
        for _ in 0..127 {
            nested = JsonValue::Array(vec![nested]);
        }
        let metadata = object_from_pairs([("nested", nested)]);
        col.insert_document(crate::DocumentInput {
            external_id: "deep".into(),
            title: None,
            source_uri: None,
            metadata: metadata.clone(),
            chunks: vec![crate::ChunkInput {
                chunk_external_id: "deep#0".into(),
                ordinal: 0,
                text: "payload".into(),
                metadata,
                source_span: None,
            }],
        })
        .unwrap();
        drop((col, db));
        for (command, flags) in [
            (
                "get-document",
                vec!["--external-id", "deep", "--include-chunks"],
            ),
            ("get-chunk", vec!["--chunk-external-id", "deep#0"]),
        ] {
            let mut args = vec![
                "coffer".into(),
                command.into(),
                path.display().to_string(),
                "--collection".into(),
                "notes".into(),
            ];
            args.extend(flags.into_iter().map(String::from));
            let (output, exit_code) = run_with_exit_code(args).unwrap();
            assert_eq!(exit_code, 0);
            assert_eq!(output.matches("depth-boundary").count(), 2);
            assert!(output.contains("\"ok\": true"));
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn delete_and_drop_commands_report_public_result_schema() {
        let (path, _document_id, chunk_id) = create_cli_schema_fixture("delete-chunk-schema");
        let output = run(vec![
            "coffer".into(),
            "delete-chunk".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            chunk_id,
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("chunk_external_id").and_then(JsonValue::as_str),
            Some("doc#1")
        );
        assert_eq!(
            result.get("embedding_deleted").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert!(result
            .get("tombstone_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .starts_with("tombstone:chunk:"));
        let _ = std::fs::remove_file(path);

        let (path, document_id, _chunk_id) = create_cli_schema_fixture("delete-document-schema");
        let output = run(vec![
            "coffer".into(),
            "delete-document".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--document-id".into(),
            document_id,
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("external_id").and_then(JsonValue::as_str),
            Some("doc")
        );
        assert_eq!(
            result.get("chunks_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("embeddings_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert!(result
            .get("tombstone_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .starts_with("tombstone:document:"));
        let _ = std::fs::remove_file(path);

        let (path, _document_id, _chunk_id) = create_cli_schema_fixture("drop-schema");
        let output = run(vec![
            "coffer".into(),
            "drop-collection".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("documents_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("chunks_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("embeddings_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert!(result
            .get("tombstone_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .starts_with("tombstone:collection:"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn why_not_output_uses_success_envelope_and_query_context() {
        let path = temp_path("why-not-envelope");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "wal recovery cli why not".into(),
                ordinal: 1,
                metadata: parse(r#"{"lang":"python"}"#).unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        drop(col);
        drop(db);

        let output = run(vec![
            "coffer".into(),
            "why-not".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--query".into(),
            "wal recovery".into(),
            "--chunk-id".into(),
            "doc#1".into(),
            "--filter".into(),
            r#"{"chunk.lang":"rust"}"#.into(),
            "--weights".into(),
            r#"{"bm25":1.0,"vector":0.0,"metadata":0.0}"#.into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("why-not")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        let context = result
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            context.get("weights_source").and_then(JsonValue::as_str),
            Some("user_override")
        );
        let expected = result
            .get("expected_chunk")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            expected
                .get("chunk_external_id")
                .and_then(JsonValue::as_str),
            Some("doc#1")
        );
        let codes = result
            .get("diagnosis_codes")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert!(codes
            .iter()
            .any(|code| code.as_str() == Some("filtered_by_metadata")));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn stats_include_index_when_requested() {
        let path = temp_path("stats-include-index");
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

        let output = run(vec![
            "coffer".into(),
            "stats".into(),
            path.display().to_string(),
            "--include-index".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(
            result.get("format_version"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("file_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("data_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("index_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert_eq!(
            result
                .get("background_compaction_enabled")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        let index_stats = result
            .get("index_stats")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            index_stats
                .get("rebuild_needed")
                .and_then(JsonValue::as_bool),
            Some(false)
        );

        let output = run(vec![
            "coffer".into(),
            "collection-stats".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--include-index".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(
            result.get("format_version"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("data_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("index_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert_eq!(
            result
                .get("background_compaction_enabled")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        assert!(matches!(
            result.get("index_stats"),
            Some(JsonValue::Object(_))
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compact_output_uses_public_size_aliases() {
        let path = temp_path("compact-schema");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let output = run(vec![
            "coffer".into(),
            "compact".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(
            result.get("before_file_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("after_file_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("records_removed"),
            Some(JsonValue::Number(_))
        ));
        let output = run(vec![
            "coffer".into(),
            "compact".into(),
            path.display().to_string(),
            "--rebuild-indexes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("compact")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn import_commands_record_effective_durability() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("cofferdb-cli-durability-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("durability.coffer");
        std::fs::write(
            dir.join("documents.jsonl"),
            r#"{"external_id":"doc","metadata":{}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("chunks.jsonl"),
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"durability import","metadata":{}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_model":null,"embedding_dim":2,"distance_metric":"cosine"}"#,
        )
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "import".into(),
            db_path.display().to_string(),
            "--manifest".into(),
            dir.join("manifest.json").display().to_string(),
            "--durability".into(),
            "fast".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let envelope_warnings = json.get("warnings").and_then(JsonValue::as_array).unwrap();
        assert_eq!(envelope_warnings.len(), 2);
        assert_eq!(
            envelope_warnings[0].get("code").and_then(JsonValue::as_str),
            Some("missing_reproducibility_hash")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result
                .get("warnings")
                .and_then(JsonValue::as_array)
                .map(|warnings| warnings.len()),
            Some(2)
        );
        assert_eq!(
            result.get("durability").and_then(JsonValue::as_str),
            Some("fast")
        );
        assert_eq!(
            result.get("documents_imported").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("chunks_imported").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(result.get("skipped").and_then(JsonValue::as_f64), Some(0.0));
        assert_eq!(
            result.get("replaced").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(result.get("errors").and_then(JsonValue::as_f64), Some(0.0));

        let output = run(vec![
            "coffer".into(),
            "import".into(),
            db_path.display().to_string(),
            "--manifest".into(),
            dir.join("manifest.json").display().to_string(),
            "--on-conflict".into(),
            "replace".into(),
            "--durability".into(),
            "fast".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("replaced").and_then(JsonValue::as_f64),
            Some(2.0)
        );

        std::fs::write(
            dir.join("embeddings.jsonl"),
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        )
        .unwrap();
        let output = run(vec![
            "coffer".into(),
            "import-embeddings".into(),
            db_path.display().to_string(),
            dir.join("embeddings.jsonl").display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--on-conflict".into(),
            "replace".into(),
            "--durability".into(),
            "fast".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("durability").and_then(JsonValue::as_str),
            Some("fast")
        );
        assert_eq!(
            result
                .get("embeddings_imported")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("replaced").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(
            result.get("missing_chunks").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(
            result.get("model_mismatches").and_then(JsonValue::as_f64),
            Some(0.0)
        );

        let output = run(vec![
            "coffer".into(),
            "import-embeddings".into(),
            db_path.display().to_string(),
            dir.join("embeddings.jsonl").display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--durability".into(),
            "fast".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("replaced").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn profile_help_reports_exact_names_and_candidate_limits_without_aliases() {
        let output = run(vec!["coffer".into(), "profile-help".into()]).unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("default").and_then(JsonValue::as_str),
            Some("balanced")
        );
        let profiles = result
            .get("profiles")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert_eq!(profiles.len(), 3);
        for (profile, (name, candidates, cache)) in profiles.iter().zip([
            ("expanded", 500.0, 512.0),
            ("balanced", 200.0, 256.0),
            ("constrained", 50.0, 64.0),
        ]) {
            assert_eq!(profile.get("cli").and_then(JsonValue::as_str), Some(name));
            assert_eq!(profile.get("json").and_then(JsonValue::as_str), Some(name));
            assert_eq!(profile.get("alias"), Some(&JsonValue::Null));
            let description = profile
                .get("description")
                .and_then(JsonValue::as_str)
                .unwrap();
            let scenarios: &[&str] = match name {
                "expanded" => &[
                    "retrieval experiments",
                    "candidate inspection",
                    "quality evaluation",
                ],
                "balanced" => &["everyday local RAG", "code search"],
                "constrained" => &["limited host resources"],
                _ => unreachable!(),
            };
            for scenario in scenarios {
                assert!(
                    description.contains(scenario),
                    "{name} help must describe the applicable scenario: {scenario}"
                );
            }
            for field in ["bm25", "vector", "fusion"] {
                assert_eq!(
                    profile
                        .get("candidate_limits")
                        .unwrap()
                        .get(field)
                        .and_then(JsonValue::as_f64),
                    Some(candidates)
                );
            }
            assert_eq!(
                profile.get("cache_budget_mb").and_then(JsonValue::as_f64),
                Some(cache)
            );
        }
    }

    #[test]
    fn invalid_profile_and_durability_report_machine_readable_detail() {
        let profile_err = run(vec![
            "coffer".into(),
            "search".into(),
            "notes.coffer".into(),
            "wal".into(),
            "--collection".into(),
            "notes".into(),
            "--profile".into(),
            "turbo".into(),
        ])
        .unwrap_err();
        assert_eq!(profile_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&profile_err, "field"), Some("--profile"));
        assert_eq!(detail_value(&profile_err, "actual"), Some("turbo"));
        assert_eq!(
            detail_value(&profile_err, "expected"),
            Some("expanded|balanced|constrained|help")
        );

        let durability_err = run(vec![
            "coffer".into(),
            "benchmark".into(),
            "notes.coffer".into(),
            "--dataset".into(),
            "dataset".into(),
            "--durability".into(),
            "unsafe".into(),
        ])
        .unwrap_err();
        assert_eq!(durability_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&durability_err, "field"), Some("--durability"));
        assert_eq!(detail_value(&durability_err, "actual"), Some("unsafe"));
        assert_eq!(
            detail_value(&durability_err, "expected"),
            Some("commit|fast")
        );
    }

    #[test]
    fn invalid_top_k_reports_machine_readable_detail() {
        let search_err = run(vec![
            "coffer".into(),
            "search".into(),
            "notes.coffer".into(),
            "wal".into(),
            "--collection".into(),
            "notes".into(),
            "--top-k".into(),
            "many".into(),
        ])
        .unwrap_err();
        assert_eq!(search_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&search_err, "field"), Some("--top-k"));
        assert_eq!(detail_value(&search_err, "actual"), Some("many"));
        assert_eq!(
            detail_value(&search_err, "expected"),
            Some("positive_integer")
        );

        let why_not_err = run(vec![
            "coffer".into(),
            "why-not".into(),
            "notes.coffer".into(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            "chunk_1".into(),
            "--top-k".into(),
            "ten".into(),
        ])
        .unwrap_err();
        assert_eq!(why_not_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&why_not_err, "field"), Some("--top-k"));
        assert_eq!(detail_value(&why_not_err, "actual"), Some("ten"));
        assert_eq!(
            detail_value(&why_not_err, "expected"),
            Some("positive_integer")
        );
    }

    #[test]
    fn help_reports_public_cli_surface_and_flags() {
        let output = run(vec!["coffer".into(), "help".into()]).unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("help")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        let commands = result
            .get("commands")
            .and_then(JsonValue::as_array)
            .unwrap();
        for name in [
            "init",
            "create-collection",
            "search",
            "vector-search",
            "stats",
            "check",
            "compact",
            "benchmark",
            "export",
            "snapshot",
        ] {
            assert!(
                commands
                    .iter()
                    .any(|command| command.get("name").and_then(JsonValue::as_str) == Some(name)),
                "missing help command {name}"
            );
        }
        let search = help_command_by_name(commands, "search");
        assert_eq!(
            search.get("synopsis").and_then(JsonValue::as_str),
            Some(
                format!(
                    "coffer search <path> [<query>] [--top-k <1..={}>]",
                    crate::query::MAX_SEARCH_TOP_K
                )
                .as_str()
            )
        );
        for flag in [
            "--collection",
            "--top-k",
            "--profile",
            "--filter",
            "--weights",
            "--metadata-boost",
            "--embedding",
        ] {
            assert!(help_command_has_flag(search, flag), "search missing {flag}");
        }
        let create_collection = help_command_by_name(commands, "create-collection");
        for flag in [
            "--dim",
            "--metric",
            "--embedding-model",
            "--vectors-normalized",
        ] {
            assert!(
                help_command_has_flag(create_collection, flag),
                "create-collection missing {flag}"
            );
        }
        let benchmark = help_command_by_name(commands, "benchmark");
        for flag in [
            "--dataset",
            "--profile",
            "--durability",
            "--output",
            "--result-file",
            "--require-baseline",
        ] {
            assert!(
                help_command_has_flag(benchmark, flag),
                "benchmark missing {flag}"
            );
        }
        let import = help_command_by_name(commands, "import");
        for flag in ["--manifest", "--on-conflict", "--durability"] {
            assert!(help_command_has_flag(import, flag), "import missing {flag}");
        }
        for flag in ["--collection", "--format", "--documents"] {
            assert!(
                !help_command_has_flag(import, flag),
                "import should not advertise unimplemented convenience flag {flag}"
            );
        }
        let check = help_command_by_name(commands, "check");
        assert!(help_command_has_flag(check, "--verify-indexes"));
        let compact = help_command_by_name(commands, "compact");
        assert!(help_command_has_flag(compact, "--rebuild-indexes"));
        let rebuild_index = help_command_by_name(commands, "rebuild-index");
        assert_eq!(
            rebuild_index.get("synopsis").and_then(JsonValue::as_str),
            Some("coffer rebuild-index <path> --collection <name>")
        );
        assert!(help_command_has_flag(rebuild_index, "--collection"));
        let import_embeddings = help_command_by_name(commands, "import-embeddings");
        assert!(help_command_has_flag(import_embeddings, "--on-conflict"));
        assert_eq!(
            help_command_by_name(commands, "why-not")
                .get("stability")
                .and_then(JsonValue::as_str),
            Some("debug")
        );
        assert_eq!(
            help_command_by_name(commands, "snapshot")
                .get("stability")
                .and_then(JsonValue::as_str),
            Some("future")
        );
        let exit_codes = result
            .get("exit_codes")
            .and_then(JsonValue::as_array)
            .unwrap();
        for (code, meaning) in [
            (0.0, "success"),
            (1.0, "user_or_input_error"),
            (2.0, "file_or_corruption_error"),
            (3.0, "check_issues"),
            (4.0, "lock_contention"),
        ] {
            assert!(
                exit_codes.iter().any(|entry| {
                    entry.get("code").and_then(JsonValue::as_f64) == Some(code)
                        && entry.get("meaning").and_then(JsonValue::as_str) == Some(meaning)
                }),
                "missing exit code {code} {meaning}"
            );
        }
    }

    fn help_command_by_name<'a>(commands: &'a [JsonValue], name: &str) -> &'a JsonValue {
        commands
            .iter()
            .find(|command| command.get("name").and_then(JsonValue::as_str) == Some(name))
            .unwrap_or_else(|| panic!("missing help command {name}"))
    }

    fn help_command_has_flag(command: &JsonValue, flag: &str) -> bool {
        command
            .get("flags")
            .and_then(JsonValue::as_array)
            .map(|flags| {
                flags
                    .iter()
                    .any(|value| value.as_str().map(|value| value == flag).unwrap_or(false))
            })
            .unwrap_or(false)
    }

    #[test]
    fn weights_require_all_fields() {
        let err = parse_weights(r#"{"bm25":1.0}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert!(err.message().contains("numeric vector"));
        assert_eq!(detail_value(&err, "field"), Some("weights.vector"));
        assert_eq!(detail_value(&err, "expected"), Some("number"));
    }

    #[test]
    fn weights_must_sum_to_one() {
        let err = parse_weights(r#"{"bm25":1.0,"vector":1.0,"metadata":0.0}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert!(err.message().contains("sum to 1.0"));
        assert_eq!(detail_value(&err, "field"), Some("weights"));
        assert_eq!(detail_value(&err, "reason"), Some("sum_mismatch"));
        assert_eq!(detail_value(&err, "expected_sum"), Some("1"));
        assert_eq!(detail_value(&err, "actual_sum"), Some("2"));
    }

    #[test]
    fn weights_reject_unknown_fields_with_machine_readable_detail() {
        let err =
            parse_weights(r#"{"bm25":1.0,"vector":0.0,"metadata":0.0,"debug":true}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert!(err.message().contains("unsupported field debug"));
        assert_eq!(detail_value(&err, "field"), Some("weights.debug"));
        assert_eq!(detail_value(&err, "reason"), Some("unsupported_field"));
    }

    #[test]
    fn vector_search_rejects_weights_and_metadata_boost() {
        let missing_embedding = run(vec![
            "coffer".into(),
            "vector-search".into(),
            "notes.coffer".into(),
            "--collection".into(),
            "notes".into(),
        ])
        .unwrap_err();
        assert_eq!(missing_embedding.code(), ErrorCode::CliUsage);
        assert!(missing_embedding.message().contains("--embedding"));

        let err = run(vec![
            "coffer".into(),
            "vector-search".into(),
            "notes.coffer".into(),
            "--collection".into(),
            "notes".into(),
            "--weights".into(),
            r#"{"bm25":0.0,"vector":1.0,"metadata":0.0}"#.into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("--weights"));
        assert_eq!(detail_value(&err, "command"), Some("vector-search"));
        assert_eq!(detail_value(&err, "field"), Some("--weights"));
        assert_eq!(detail_value(&err, "reason"), Some("unsupported_option"));

        let err = run(vec![
            "coffer".into(),
            "vector-search".into(),
            "notes.coffer".into(),
            "--collection".into(),
            "notes".into(),
            "--metadata-boost".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("--metadata-boost"));
        assert_eq!(detail_value(&err, "command"), Some("vector-search"));
        assert_eq!(detail_value(&err, "field"), Some("--metadata-boost"));
        assert_eq!(detail_value(&err, "reason"), Some("unsupported_option"));
    }

    #[test]
    fn query_embedding_file_requires_public_object_schema() {
        let path = temp_path("query-embedding-schema");

        std::fs::write(&path, r#"[1.0,0.0]"#).unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);

        std::fs::write(&path, r#"{"dim":2,"vector":[1.0,0.0]}"#).unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert!(err.message().contains("model_name"));

        std::fs::write(&path, r#"{"model_name":"m","dim":2,"vector":[1.0]}"#).unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("1"));

        std::fs::write(&path, r#"{"model_name":"m","dim":2,"vector":[1.0,"bad"]}"#).unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "field"), Some("vector"));

        std::fs::write(
            &path,
            r#"{"model_name":"m","dim":2,"vector":[1.0,0.0],"chunk_external_id":"unexpected"}"#,
        )
        .unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("chunk_external_id"));

        std::fs::write(&path, r#"{"model_name":"m","dim":2,"vector":[1.0,0.0]}"#).unwrap();
        let input = read_embedding_file(&path.display().to_string()).unwrap();
        assert_eq!(input.model_name.as_deref(), Some("m"));
        assert_eq!(input.dim, 2);
        assert_eq!(input.vector, vec![1.0, 0.0]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn search_accepts_embedding_without_query_text() {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "cofferdb-cli-queryless-search-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("queryless.coffer");
        let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "queryless vector search".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![crate::model::EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        drop(db);
        let embedding_path = dir.join("query_embedding.json");
        std::fs::write(
            &embedding_path,
            r#"{"model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        )
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "search".into(),
            db_path.display().to_string(),
            "--embedding".into(),
            embedding_path.display().to_string(),
            "--collection".into(),
            "notes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        let context = result
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            context.get("text_present").and_then(JsonValue::as_bool),
            Some(false)
        );
        assert_eq!(
            context
                .get("embedding_present")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        let candidate_counts = result
            .get("candidate_counts")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert!(matches!(
            candidate_counts.get("vector"),
            Some(JsonValue::Number(_))
        ));
        let results = result.get("results").and_then(JsonValue::as_array).unwrap();
        assert_eq!(
            results[0]
                .get("chunk_external_id")
                .and_then(JsonValue::as_str),
            Some("doc#1")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn error_output_uses_failure_envelope() {
        let err = Error::new(ErrorCode::CliUsage, "missing command");
        let output = error_output(None, &err);
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("unknown")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(false));
        let error = json.get("error").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            error.get("code").and_then(JsonValue::as_str),
            Some("cli_usage")
        );
        assert_eq!(
            error.get("message").and_then(JsonValue::as_str),
            Some("missing command")
        );
        assert!(matches!(error.get("detail"), Some(JsonValue::Object(_))));
    }

    #[test]
    fn resource_exhaustion_preserves_the_failure_envelope_and_exact_details() {
        let error = Error::resource_exhausted(
            "temporary_disk",
            "temporary_staging",
            u64::MAX,
            u128::from(u64::MAX) + 1,
        );
        let output = parse(&error_output(Some("import"), &error)).unwrap();
        assert_eq!(
            output.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        assert_eq!(
            output.get("command").and_then(JsonValue::as_str),
            Some("import")
        );
        assert_eq!(output.get("ok").and_then(JsonValue::as_bool), Some(false));
        let error = output.get("error").unwrap();
        crate::test_error_contract::validate_error(error).unwrap();
        assert_eq!(
            error.get("code").and_then(JsonValue::as_str),
            Some("resource_exhausted")
        );
        assert_eq!(error.get("detail").unwrap(), &parse(r#"{"resource":"temporary_disk","operation":"temporary_staging","limit_bytes":"18446744073709551615","requested_bytes":"18446744073709551616"}"#).unwrap());
    }

    #[test]
    fn cli_rejects_unexpected_arguments() {
        let path = temp_path("unexpected-argument");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let err = run(vec![
            "coffer".into(),
            "stats".into(),
            path.display().to_string(),
            "--mystery".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("unexpected argument --mystery"));
        assert_eq!(detail_value(&err, "field"), Some("argument"));
        assert_eq!(detail_value(&err, "actual"), Some("--mystery"));
        assert_eq!(detail_value(&err, "reason"), Some("unexpected_argument"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unknown_command_reports_machine_readable_detail() {
        let err = run(vec!["coffer".into(), "frobnicate".into()]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("unknown command frobnicate"));
        assert_eq!(detail_value(&err, "field"), Some("command"));
        assert_eq!(detail_value(&err, "actual"), Some("frobnicate"));
        assert_eq!(detail_value(&err, "reason"), Some("unknown_command"));
    }

    #[test]
    fn missing_flag_value_reports_machine_readable_detail() {
        let err = run(vec![
            "coffer".into(),
            "search".into(),
            "notes.coffer".into(),
            "wal".into(),
            "--collection".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("--collection requires a value"));
        assert_eq!(detail_value(&err, "field"), Some("--collection"));
        assert_eq!(detail_value(&err, "reason"), Some("missing_value"));
    }

    #[test]
    fn missing_required_argument_reports_machine_readable_detail() {
        let err = run(vec!["coffer".into(), "stats".into()]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("path is required"));
        assert_eq!(detail_value(&err, "field"), Some("path"));
        assert_eq!(
            detail_value(&err, "reason"),
            Some("missing_required_argument")
        );
    }

    #[test]
    fn export_accepts_explicit_jsonl_format() {
        let path = temp_path("export-format");
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
        let mut out = std::env::temp_dir();
        out.push(format!("cofferdb-cli-export-format-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let output = run(vec![
            "coffer".into(),
            "export".into(),
            path.display().to_string(),
            out.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--format".into(),
            "jsonl".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("format").and_then(JsonValue::as_str),
            Some("jsonl")
        );
        assert_eq!(
            result.get("documents_exported").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(
            result.get("manifest").and_then(JsonValue::as_str),
            Some("export_manifest.json")
        );
        assert!(out.join("export_manifest.json").exists());
        let err = run(vec![
            "coffer".into(),
            "export".into(),
            path.display().to_string(),
            out.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--format".into(),
            "csv".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("--format must be jsonl"));
        assert_eq!(detail_value(&err, "field"), Some("--format"));
        assert_eq!(detail_value(&err, "actual"), Some("csv"));
        assert_eq!(detail_value(&err, "expected"), Some("jsonl"));
        let _ = std::fs::remove_dir_all(out);
        let _ = std::fs::remove_file(path);
    }
}
