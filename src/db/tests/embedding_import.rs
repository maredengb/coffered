use super::*;

fn fixture(label: &str, chunks: usize) -> (PathBuf, Database, CollectionHandle) {
    fixture_with_cache(label, chunks, 1)
}

fn fixture_with_cache(
    label: &str,
    chunks: usize,
    cache: usize,
) -> (PathBuf, Database, CollectionHandle) {
    let path = temp_path(label);
    let db = Database::create(
        &path,
        DatabaseConfig {
            resource_profile: ResourceProfile::Constrained,
            cache_budget_mb: Some(cache),
            ..DatabaseConfig::default()
        },
    )
    .unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "document".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..chunks)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("c{ordinal}"),
                    ordinal,
                    text: "topic retrieval".into(),
                    metadata: JsonValue::object(),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
    (path, db, collection)
}

fn row(id: &str) -> String {
    format!("{{\"chunk_external_id\":\"{id}\",\"model_name\":\"m\",\"dim\":2,\"vector\":[0,1]}}\n")
}

#[test]
fn embedding_update_preflight_rejects_bad_tail_before_any_commit() {
    let (path, db, collection) = fixture("embedding-update-preflight", 2);
    collection
        .put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "c0".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
    let input = temp_path("embedding-update-preflight-input");
    let before = fs::read(&path).unwrap();
    let context = db.json_context();
    let available = context.available_workspace_bytes();
    for (tail, code) in [
        ("not json\n".into(), ErrorCode::InvalidJsonlSchema),
        (row("missing"), ErrorCode::ChunkNotFound),
        (row("c0"), ErrorCode::DuplicateExternalId),
        (
            row("c1").replace("\"m\"", "\"other\""),
            ErrorCode::EmbeddingModelMismatch,
        ),
        (
            row("c1").replace("[0,1]", "[1]"),
            ErrorCode::InvalidEmbeddingDimension,
        ),
    ] {
        fs::write(&input, row("c0") + &tail).unwrap();
        let error = db
            .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Replace)
            .unwrap_err();
        assert_eq!(error.code(), code);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "path" && value == &input.display().to_string()));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(collection.stats().unwrap().embedding_count, 1);
        assert_eq!(db.temporary_work.used(), 0);
        assert_eq!(context.available_workspace_bytes(), available);
    }
    let result = collection
        .search()
        .query_embedding(QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        })
        .execute()
        .unwrap();
    assert_eq!(result.results[0].vector_score, Some(1.0));
    drop((collection, db));
    fs::remove_file(path).unwrap();
    fs::remove_file(input).unwrap();
}

#[test]
fn embedding_update_success_skip_and_replace_preserve_existing_semantics() {
    let (path, db, collection) = fixture("embedding-update-conflicts", 2);
    let input = temp_path("embedding-update-conflicts-input");
    fs::write(&input, row("c0") + &row("c1")).unwrap();
    let summary = db
        .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Error)
        .unwrap();
    assert_eq!(
        (
            summary.embeddings_inserted,
            summary.embeddings_skipped,
            summary.embeddings_replaced
        ),
        (2, 0, 0)
    );
    let before = fs::read(&path).unwrap();
    let error = db
        .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Error)
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::DuplicateExternalId);
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::write(&input, row("c0") + &row("missing")).unwrap();
    let error = db
        .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Skip)
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::ChunkNotFound);
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::write(&input, row("c0") + &row("c1")).unwrap();
    let summary = db
        .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Skip)
        .unwrap();
    assert_eq!(
        (
            summary.embeddings_inserted,
            summary.embeddings_skipped,
            summary.embeddings_replaced
        ),
        (0, 2, 0)
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::write(&input, row("c0").replace("[0,1]", "[1,0]")).unwrap();
    let summary = db
        .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Replace)
        .unwrap();
    assert_eq!(
        (summary.embeddings_inserted, summary.embeddings_replaced),
        (1, 1)
    );
    let result = collection
        .search()
        .query_embedding(QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        })
        .execute()
        .unwrap();
    assert_eq!(result.results[0].chunk_external_id, "c0");
    assert_eq!(result.results[0].vector_score, Some(1.0));
    assert_eq!(collection.stats().unwrap().embedding_count, 2);
    drop((collection, db));
    assert_eq!(
        Database::open_read_only(&path)
            .unwrap()
            .collection("notes")
            .unwrap()
            .stats()
            .unwrap()
            .embedding_count,
        2
    );
    fs::remove_file(path).unwrap();
    fs::remove_file(input).unwrap();
}

#[test]
fn embedding_update_uses_shared_budget_and_failure_preserves_busy_owner() {
    let (path, db, collection) = fixture("embedding-update-shared", 2);
    let input = temp_path("embedding-update-shared-input");
    fs::write(&input, row("c0")).unwrap();
    let before = fs::read(&path).unwrap();
    let context = db.json_context();
    let mut busy = context.workspace_reservation();
    let available = context.available_workspace_bytes();
    busy.grow_to(available - 1, available - 1, "operation_scratch")
        .unwrap();
    let error = db
        .clone()
        .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Replace)
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(busy.reserved_bytes(), available - 1);
    assert_eq!(context.available_workspace_bytes(), 1);
    assert_eq!(db.temporary_work.used(), 0);
    drop(busy);
    assert_eq!(context.available_workspace_bytes(), available);
    assert_eq!(
        db.import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Replace)
            .unwrap()
            .embeddings_inserted,
        1
    );
    drop((collection, db));
    fs::remove_file(path).unwrap();
    fs::remove_file(input).unwrap();
}

#[test]
fn embedding_update_batch_failure_retains_only_committed_prefix() {
    let (path, db, collection) = fixture("embedding-update-prefix", 751);
    let input = temp_path("embedding-update-prefix-input");
    fs::write(
        &input,
        (0..751).map(|i| row(&format!("c{i}"))).collect::<String>(),
    )
    .unwrap();
    let before_commit = db.stats().unwrap().last_commit_id;
    db.lock().unwrap().import_fail_after_batches = Some(1);
    let error = db
        .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Replace)
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::IoError);
    let prefix = collection.stats().unwrap().embedding_count;
    assert!(prefix > 0 && prefix <= 250);
    assert_eq!(db.stats().unwrap().last_commit_id, before_commit + 1);
    assert_eq!(db.temporary_work.used(), 0);
    assert_eq!(
        Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true
            }
        )
        .unwrap()
        .last_valid_commit,
        before_commit + 1
    );
    drop((collection, db));
    let db = Database::open(&path).unwrap();
    let collection = db.collection("notes").unwrap();
    assert_eq!(collection.stats().unwrap().embedding_count, prefix);
    let summary = db
        .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Skip)
        .unwrap();
    assert_eq!(
        (summary.embeddings_inserted, summary.embeddings_skipped),
        (751 - prefix, prefix)
    );
    assert_eq!(collection.stats().unwrap().embedding_count, 751);
    assert!(!db.index_stats().unwrap().rebuild_needed);
    drop((collection, db));
    fs::remove_file(path).unwrap();
    fs::remove_file(input).unwrap();
}

#[test]
fn embedding_update_long_ids_are_not_limited_by_the_sort_batch_preference() {
    for cache in [0, 1] {
        let (path, db, collection) = fixture_with_cache("embedding-update-long-id", 0, cache);
        let id = "x".repeat(600 * 1024);
        collection
            .insert_document(DocumentInput {
                external_id: "long-document".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: id.clone(),
                    ordinal: 0,
                    text: "topic".into(),
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            })
            .unwrap();
        let input = temp_path("embedding-update-long-input");
        fs::write(&input, row(&id)).unwrap();
        assert_eq!(
            db.import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Error)
                .unwrap()
                .embeddings_inserted,
            1
        );
        let before = fs::read(&path).unwrap();
        fs::write(&input, row(&id) + &row(&id)).unwrap();
        let error = db
            .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Replace)
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::DuplicateExternalId);
        assert_eq!(error_detail_value(&error, "first_line"), Some("1"));
        assert_eq!(error_detail_value(&error, "line"), Some("2"));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(collection.stats().unwrap().embedding_count, 1);
        assert_eq!(db.temporary_work.used(), 0);
        drop((collection, db));
        fs::remove_file(input).unwrap();
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn embedding_update_reclaims_resident_cache_at_sort_and_line_admission() {
    let path = cached_vector_batch_fixture("embedding-update-cache-reclaim");
    let input = temp_path("embedding-update-cache-input");
    fs::write(&input, " ".repeat(8192) + &row("chunk-1")).unwrap();
    let before = fs::read(&path).unwrap();
    for headroom in [60 * 1024, 75 * 1024] {
        let db = Database::open(&path).unwrap();
        let collection = db.collection("first").unwrap();
        let context = db.json_context();
        let (_, _, cached) = super::query_preparation::cache_for_preparation(&db);
        let held = context
            .cache_reservation(
                (context.available_total_bytes() - headroom) as u128,
                usize::MAX,
            )
            .unwrap();
        let summary = db
            .import_embeddings_jsonl(collection.collection_id(), &input, OnConflict::Skip)
            .unwrap();
        assert_eq!(
            (summary.embeddings_inserted, summary.embeddings_skipped),
            (0, 1)
        );
        assert_eq!(db.lock().unwrap().optional_cache_bytes(), 0);
        assert_eq!(context.available_total_bytes(), headroom + cached);
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(db.temporary_work.used(), 0);
        drop((held, collection, db));
    }
    fs::remove_file(input).unwrap();
    fs::remove_file(path).unwrap();
}

#[test]
fn prepared_embedding_batch_reports_real_working_limit_before_any_commit() {
    let (path, db, collection) = fixture_with_cache("embedding-update-real-batch-limit", 1, 0);
    let context = db.json_context();
    let chunk = collection.get_chunk_by_external_id("c0").unwrap();
    let mut records = crate::storage::PreparedRecordsBuilder::new_in(
        &std::env::temp_dir(),
        db.temporary_work.limit_bytes(),
        context.clone(),
        db.temporary_work.clone(),
    )
    .unwrap();
    records
        .push(RecordPayload::EmbeddingPut(Embedding {
            embedding_id: "emb_1".into(),
            chunk_id: chunk.chunk_id,
            collection_id: collection.collection_id().into(),
            chunk_external_id: "c0".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
            created_at: "1970-01-01T00:00:00Z".into(),
        }))
        .unwrap();
    let prepared = records.finish().unwrap();
    let before = fs::read(&path).unwrap();
    let mut busy = context.workspace_reservation();
    let available = context.available_workspace_bytes();
    busy.grow_to(available - 1, available - 1, "operation_scratch")
        .unwrap();
    let error = db
        .lock()
        .unwrap()
        .append_import_batches(
            prepared,
            ImportIndexBuildMode::DeferredFullCheckpoint,
            &context,
        )
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    assert_eq!(
        error_detail_value(&error, "operation"),
        Some("batch_prepare")
    );
    let limit = error_detail_value(&error, "limit_bytes")
        .unwrap()
        .parse::<u128>()
        .unwrap();
    let requested = error_detail_value(&error, "requested_bytes")
        .unwrap()
        .parse::<u128>()
        .unwrap();
    assert_eq!(
        limit,
        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES as u128
    );
    assert!(requested > limit);
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(db.temporary_work.used(), 0);
    assert_eq!(busy.reserved_bytes(), available - 1);
    drop(busy);
    assert_eq!(collection.stats().unwrap().embedding_count, 0);
    drop((collection, db));
    fs::remove_file(path).unwrap();
}
