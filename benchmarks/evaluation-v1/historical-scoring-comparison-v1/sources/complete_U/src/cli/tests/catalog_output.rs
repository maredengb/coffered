use super::*;
use crate::json::JsonProcessingPolicyV1;

const NAME_BYTES: usize = 65_539;
const MODEL_BYTES: usize = 32_771;

fn fixture(label: &str) -> PathBuf {
    let path = temp_path(label);
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    for (name, embedding_model) in [
        ("plain".to_owned(), None),
        ("\"".repeat(NAME_BYTES), Some("m".repeat(MODEL_BYTES))),
    ] {
        db.create_collection(CollectionConfig {
            name,
            embedding_model,
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    }
    path
}

#[test]
fn catalog_commands_keep_credit_after_database_close_and_preserve_wire() {
    let path = fixture("catalog-command-credit");
    let original = fs::read(&path).unwrap();
    let name = "\"".repeat(NAME_BYTES);
    for (command, collection, include_index) in [
        ("list-collections", None, false),
        ("stats", None, false),
        ("stats", None, true),
        ("collection-stats", Some("plain"), false),
        ("collection-stats", Some(name.as_str()), true),
    ] {
        let expected = {
            let db = Database::open_read_only(&path).unwrap();
            let mut value = match command {
                "list-collections" => object_from_pairs([(
                    "collections",
                    JsonValue::Array(
                        db.list_collections()
                            .unwrap()
                            .iter()
                            .map(|info| info.to_json())
                            .collect(),
                    ),
                )]),
                "collection-stats" => db
                    .collection(collection.unwrap())
                    .unwrap()
                    .stats()
                    .unwrap()
                    .to_json(),
                _ => db.stats().unwrap().to_json(),
            };
            if include_index {
                insert_object_field(
                    &mut value,
                    "index_stats",
                    db.index_stats().unwrap().to_json(),
                );
            }
            cli_envelope(command, true, value, None, 7)
        };
        let mut args = vec![path.display().to_string()];
        if let Some(collection) = collection {
            args.extend(["--collection".into(), collection.to_owned()]);
        }
        if include_index {
            args.push("--include-index".into());
        }
        let result = run_command(command, Args::new(args)).unwrap();
        let expected_keys: &[&str] = match command {
            "list-collections" => &["collections"],
            "collection-stats" => &[
                "active_profile",
                "background_compaction_enabled",
                "cache_budget_mb",
                "chunk_count",
                "collection",
                "data_size_bytes",
                "dead_bytes_estimate",
                "deleted_count",
                "document_count",
                "embedding_count",
                "format_version",
                "index_size_bytes",
                "last_commit_id",
                "live_bytes",
                "pending_compaction_bytes",
                "resource_profile",
                "text_index_segments",
                "text_index_terms",
                "tokenizer",
                "tombstone_count",
                "vector_count",
            ],
            _ => &[
                "active_profile",
                "background_compaction_enabled",
                "cache_budget_mb",
                "chunk_count",
                "collection_count",
                "data_size_bytes",
                "dead_bytes_estimate",
                "deleted_count",
                "document_count",
                "embedding_count",
                "file_size",
                "file_size_bytes",
                "format_version",
                "index_size_bytes",
                "last_commit_id",
                "live_bytes",
                "pending_compaction_bytes",
                "tombstone_count",
            ],
        };
        assert_eq!(
            result
                .value
                .as_object()
                .unwrap()
                .keys()
                .filter(|key| key.as_str() != "index_stats")
                .map(String::as_str)
                .collect::<Vec<_>>(),
            expected_keys
        );
        assert_eq!(result.value.get("index_stats").is_some(), include_index);
        if include_index {
            assert_eq!(
                result
                    .value
                    .get("index_stats")
                    .unwrap()
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                [
                    "index_size_bytes",
                    "last_index_commit_id",
                    "rebuild_needed",
                    "text_segments",
                    "text_terms",
                    "vector_count",
                    "vector_segments",
                ]
            );
        }
        if command == "list-collections" {
            for info in result.value.get("collections").unwrap().as_array().unwrap() {
                assert_eq!(
                    info.as_object()
                        .unwrap()
                        .keys()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                    [
                        "collection_id",
                        "config",
                        "created_at",
                        "distance_metric",
                        "embedding_dim",
                        "embedding_model",
                        "name",
                        "tokenizer",
                        "vector_index",
                        "vectors_normalized",
                    ]
                );
                assert_eq!(
                    info.get("vector_index")
                        .unwrap()
                        .get("type")
                        .unwrap()
                        .as_str(),
                    Some("exact")
                );
            }
        }
        let (context, work, schema) = result.retained.as_ref().unwrap();
        assert!(matches!(
            schema,
            JsonOutputSchema::Value(JsonInputKind::StoredJson)
        ));
        assert!(work.reserved_bytes() > 0);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes(),
        );
        let context = context.clone();
        assert_eq!(success_output(command, result, 7).unwrap(), expected);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn catalog_output_failures_refund_only_the_failed_stage_and_do_not_change_data() {
    let path = fixture("catalog-output-failure");
    let original = fs::read(&path).unwrap();
    let name = "\"".repeat(NAME_BYTES);
    let db = Database::open_read_only(&path).unwrap();
    let context = db.json_context();
    let previous = db.stats_json(None, true, &context).unwrap();
    let previous_value = previous.0.to_compact_string();
    let available = context.available_workspace_bytes();
    for list in [true, false] {
        let error = if list {
            crate::test_alloc::fail_byte_allocation_once(MODEL_BYTES, || {
                db.list_collections_json(&context)
            })
        } else {
            crate::test_alloc::fail_byte_allocation_once(NAME_BYTES, || {
                db.stats_json(Some(&name), true, &context)
            })
        }
        .err()
        .unwrap();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(context.available_workspace_bytes(), available);
        assert_eq!(previous.0.to_compact_string(), previous_value);
    }
    let mut busy = context.workspace_reservation();
    busy.grow_to(available, available, "operation_scratch")
        .unwrap();
    for (list, collection) in [(true, None), (false, None), (false, Some(name.as_str()))] {
        let error = if list {
            db.list_collections_json(&context)
        } else {
            db.stats_json(collection, true, &context)
        }
        .err()
        .unwrap();
        assert!(crate::json::is_memory_budget_error(&error), "{error:?}");
        assert_eq!(context.available_workspace_bytes(), 0);
        assert_eq!(previous.0.to_compact_string(), previous_value);
    }
    drop(busy);
    assert_eq!(context.available_workspace_bytes(), available);
    let body = db.stats_json(None, false, &context).unwrap();
    let body_bytes = body.1.covered_bytes();
    drop(body);
    let mut busy = context.workspace_reservation();
    busy.grow_to(
        available - body_bytes,
        available - body_bytes,
        "operation_scratch",
    )
    .unwrap();
    let body = db.stats_json(None, false, &context).unwrap();
    assert!(body.0.get("index_stats").is_none());
    drop(body);
    let error = db.stats_json(None, true, &context).err().unwrap();
    assert!(crate::json::is_memory_budget_error(&error), "{error:?}");
    assert_eq!(context.available_workspace_bytes(), body_bytes);
    assert_eq!(previous.0.to_compact_string(), previous_value);
    drop(busy);
    assert_eq!(context.available_workspace_bytes(), available);
    let (value, work) = db.stats_json(Some(&name), true, &context).unwrap();
    let occupied = context.available_workspace_bytes();
    let mut busy = context.workspace_reservation();
    busy.grow_to(occupied - 8192, occupied - 8192, "operation_scratch")
        .unwrap();
    let error = success_output(
        "collection-stats",
        CommandResult::retained(
            value,
            context.clone(),
            work,
            JsonOutputSchema::Value(JsonInputKind::StoredJson),
        ),
        7,
    )
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::RecoveryError);
    assert_eq!(
        detail_value(&error, "json_limit_reason"),
        Some("work_budget_exceeded")
    );
    drop(busy);
    assert_eq!(context.available_workspace_bytes(), available);
    assert_eq!(previous.0.to_compact_string(), previous_value);
    assert_eq!(fs::read(&path).unwrap(), original);
    drop(previous);
    drop(db);
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn collection_list_outer_container_is_not_a_new_business_json_unit() {
    let path = temp_path("catalog-list-outer-unit");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    for ordinal in 0..4 {
        db.create_collection(CollectionConfig {
            name: format!("collection-{ordinal}"),
            embedding_model: Some("m".repeat(2_250_000)),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    }
    drop(db);
    let result = run_command(
        "list-collections",
        Args::new(vec![path.display().to_string()]),
    )
    .unwrap();
    let collections = result.value.get("collections").unwrap().as_array().unwrap();
    assert_eq!(collections.len(), 4);
    assert!(collections.iter().all(|item| item
        .get("embedding_model")
        .unwrap()
        .as_str()
        .unwrap()
        .len()
        == 2_250_000));
    let expected = cli_envelope("list-collections", true, result.value.clone(), None, 0);
    let context = result.retained.as_ref().unwrap().0.clone();
    let output = success_output("list-collections", result, 0).unwrap();
    assert_eq!(output, expected);
    assert!(output.len() > 8 * 1024 * 1024);
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    fs::remove_file(path).unwrap();
}
