use super::*;
use crate::json::{JsonOperationContext, WorkLedger};

fn input(text: String) -> DocumentInput {
    DocumentInput {
        external_id: "document".into(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: vec![ChunkInput {
            chunk_external_id: "chunk".into(),
            text,
            ordinal: 0,
            metadata: JsonValue::object(),
            source_span: None,
        }],
    }
}

fn collection(db: &Database) -> CollectionHandle {
    db.create_collection(CollectionConfig {
        name: "notes".into(),
        tokenizer: Default::default(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: Some("m".into()),
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    })
    .unwrap()
}

#[test]
fn token_hint_write_refusal_and_native_failure_leave_file_commit_and_ids_unchanged() {
    let path = temp_path("token-hint-write");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = collection(&db);
    let before = fs::read(&path).unwrap();
    let commit = db.stats().unwrap().last_commit_id;
    let context = db.json_context();
    let capacity = context.available_workspace_bytes();
    let length = 131_071;
    let available = length + length / 2;
    let mut held = context.workspace_reservation();
    held.grow_to(
        capacity - available,
        capacity - available,
        "operation_scratch",
    )
    .unwrap();
    let value = input("P".repeat(length));
    let (result, allocations) =
        crate::test_alloc::key_allocations(length, || collection.insert_document(value));
    let error = result.unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted, "{error}");
    assert_eq!(
        error_detail_value(&error, "operation"),
        Some("operation_scratch")
    );
    assert_eq!(allocations.allocated, 0);
    assert_eq!(context.available_workspace_bytes(), available);
    held.retain(0);
    let value = input("P".repeat(length));
    let error =
        crate::test_alloc::fail_byte_allocation_once(length, || collection.insert_document(value))
            .unwrap_err();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(context.available_workspace_bytes(), capacity);
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(db.stats().unwrap().last_commit_id, commit);
    collection
        .insert_document(input("PROJECT foo_bar".into()))
        .unwrap();
    let document = collection.get_document_by_external_id("document").unwrap();
    let chunk = collection.get_chunk_by_external_id("chunk").unwrap();
    assert_eq!(document.document_id, "doc_0000000000000001");
    assert_eq!(chunk.chunk_id, "chk_0000000000000001");
    assert_eq!(chunk.token_count, 4);
    drop((collection, db));
    fs::remove_file(path).unwrap();
}

#[test]
fn token_hint_materialization_refusal_releases_complete_retained_chunk_on_both_axes() {
    let path = temp_path("token-hint-read");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = collection(&db);
    let length = 131_071;
    collection
        .insert_document(input("P".repeat(length)))
        .unwrap();
    let chunk = collection.get_chunk_by_external_id("chunk").unwrap();
    let before = fs::read(&path).unwrap();
    let inner = db.lock().unwrap();
    let available = length + length / 2;
    for parent in [false, true] {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
            if parent { usize::MAX } else { available },
            usize::MAX,
            if parent { available } else { usize::MAX },
        )));
        let raw = inner
            .storage
            .payload_reader()
            .unwrap()
            .read_chunk_in(
                inner.catalog.record_ref_for_chunk(&chunk.chunk_id).unwrap(),
                &context,
            )
            .unwrap();
        drop(raw);
        assert_eq!(context.available_workspace_bytes(), available);
        let error = inner
            .materialize_chunk_in(&chunk.chunk_id, &context)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted, "{error}");
        assert_eq!(
            error_detail_value(&error, "operation"),
            Some("operation_scratch")
        );
        assert_eq!(
            error_detail_value(&error, "limit_bytes"),
            Some(available.to_string().as_str())
        );
        assert_eq!(context.available_workspace_bytes(), available);
        context
            .workspace_reservation()
            .grow_to(available, available, "operation_scratch")
            .unwrap();
    }
    drop(inner);
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(
        collection.get_chunk(&chunk.chunk_id).unwrap().token_count,
        1
    );
    drop((collection, db));
    fs::remove_file(path).unwrap();
}

#[test]
fn token_hint_compact_refusal_after_cache_eviction_preserves_source_and_old_readers() {
    let directory = temp_path("token-hint-compact");
    fs::create_dir(&directory).unwrap();
    let path = directory.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = collection(&db);
    let length = 524_289;
    collection
        .insert_document(input("P".repeat(length)))
        .unwrap();
    collection
        .put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "chunk".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
    let query = |database: &Database| {
        database
            .collection("notes")
            .unwrap()
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .execute()
            .unwrap()
            .results
            .into_iter()
            .map(|hit| (hit.chunk_external_id, hit.final_score.to_bits()))
            .collect::<Vec<_>>()
    };
    let expected = query(&db);
    assert_eq!(expected, vec![("chunk".into(), 1.0f64.to_bits())]);
    {
        let inner = db.lock().unwrap();
        assert!(
            inner.lazy_vector_segment.is_some()
                || inner
                    .persistent_vector_segments
                    .as_ref()
                    .is_some_and(|cache| !cache.is_empty())
        );
    }
    let old_reader = Database::open_read_only(&path).unwrap();
    let before = fs::read(&path).unwrap();
    let commit = db.stats().unwrap().last_commit_id;
    let entries = fs::read_dir(&directory).unwrap().count();
    let context = db.json_context();
    let available = length + length / 2 + 65_536;
    let capacity = context.available_workspace_bytes();
    let mut held = context.workspace_reservation();
    held.grow_to(
        capacity - available,
        capacity - available,
        "operation_scratch",
    )
    .unwrap();
    {
        let inner = db.lock().unwrap();
        // Reserve more than the compact writer's 64 KiB buffer and small fixed
        // owners, independently of the eventual refusal's reported request.
        let mut fixed = context.workspace_reservation();
        fixed
            .grow_to(128 * 1024, 128 * 1024, "operation_scratch")
            .unwrap();
        let chunk_id = inner
            .catalog
            .chunk_external_ids
            .get(collection.collection_id(), "chunk")
            .unwrap();
        let raw = inner
            .storage
            .payload_reader()
            .unwrap()
            .read_chunk_in(
                inner.catalog.record_ref_for_chunk(chunk_id).unwrap(),
                &context,
            )
            .unwrap();
        drop(raw);
    }
    let error = db
        .compact(CompactOptions {
            rebuild_indexes: false,
        })
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted, "{error}");
    assert_eq!(
        error_detail_value(&error, "operation"),
        Some("operation_scratch")
    );
    let requested = error_detail_value(&error, "requested_bytes")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let limit = error_detail_value(&error, "limit_bytes")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let compact_peak = requested - (limit - available);
    assert!(
        (2 * length..=2 * length + 128 * 1024).contains(&compact_peak),
        "{error}"
    );
    {
        let inner = db.lock().unwrap();
        assert!(inner.lazy_vector_segment.is_none());
        assert!(inner
            .persistent_vector_segments
            .as_ref()
            .map_or(true, |cache| cache.is_empty()));
    }
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(db.stats().unwrap().last_commit_id, commit);
    assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
    assert_eq!(db.temporary_disk_budget().used(), 0);
    assert_eq!(db.temporary_work.used(), 0);
    assert_eq!(context.available_workspace_bytes(), available);
    held.retain(0);
    assert_eq!(query(&db), expected);
    assert_eq!(query(&old_reader), expected);
    assert_eq!(query(&Database::open_read_only(&path).unwrap()), expected);
    db.compact(CompactOptions {
        rebuild_indexes: true,
    })
    .unwrap();
    assert_eq!(query(&db), expected);
    assert_eq!(query(&old_reader), expected);
    drop((old_reader, collection, db));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn token_hint_import_native_failure_discards_staged_rows_before_first_commit() {
    let directory = temp_path("token-hint-import");
    fs::create_dir(&directory).unwrap();
    let path = directory.join("data.coffer");
    let manifest = directory.join("manifest.json");
    fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
    fs::write(
        directory.join("documents.jsonl"),
        "{\"external_id\":\"document\",\"metadata\":{}}\n",
    )
    .unwrap();
    let part_bytes = 131_071;
    let word = "P".repeat(part_bytes);
    fs::write(directory.join("chunks.jsonl"), format!(
        "{{\"document_external_id\":\"document\",\"chunk_external_id\":\"chunk\",\"ordinal\":0,\"text\":\"{word}_{word}\",\"metadata\":{{}}}}\n",
    )).unwrap();
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let before = fs::read(&path).unwrap();
    let commit = db.stats().unwrap().last_commit_id;
    let context = db.json_context();
    let capacity = context.available_workspace_bytes();
    let entries = fs::read_dir(&directory).unwrap().count();
    // The whole JSON field and normalized part have distinct native targets.
    for bytes in [2 * part_bytes + 1, part_bytes] {
        let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
            db.import_manifest(&manifest, OnConflict::Error)
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError, "{error}");
        assert_eq!(error_detail_value(&error, "line"), Some("1"));
        assert_eq!(error_detail_value(&error, "json_limit_reason"), None);
        assert_eq!(error_detail_value(&error, "limit"), None);
        assert_eq!(
            error_detail_value(&error, "path"),
            directory.join("chunks.jsonl").to_str()
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(db.stats().unwrap().last_commit_id, commit);
        assert_eq!(db.stats().unwrap().collection_count, 0);
        assert_eq!(context.available_workspace_bytes(), capacity);
        assert_eq!(db.temporary_disk_budget().used(), 0);
        assert_eq!(db.temporary_work.used(), 0);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
    }
    db.import_manifest(&manifest, OnConflict::Error).unwrap();
    let chunk = db
        .collection("notes")
        .unwrap()
        .get_chunk_by_external_id("chunk")
        .unwrap();
    assert_eq!(chunk.token_count, 2);
    assert_eq!(chunk.chunk_id, "chk_0000000000000001");
    assert_eq!(db.stats().unwrap().document_count, 1);
    drop(db);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn token_hint_preprocessed_import_budget_refusal_discards_staging_without_row_replay() {
    use crate::db::import::{ImportDocument, ImportRow, ManifestImportOptions};
    let path = temp_path("token-hint-preprocessed");
    let db = Database::create(
        &path,
        DatabaseConfig {
            cache_budget_mb: Some(0),
            ..DatabaseConfig::default()
        },
    )
    .unwrap();
    let collection = collection(&db);
    let before = fs::read(&path).unwrap();
    let commit = db.stats().unwrap().last_commit_id;
    let context = db.json_context();
    let capacity = context.available_workspace_bytes();
    let options = ManifestImportOptions {
        tokenizer: Default::default(),
        collection_name: "notes".into(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: Some("m".into()),
        document_path: "documents.jsonl".into(),
        chunk_path: "chunks.jsonl".into(),
        embedding_path: None,
        on_conflict: OnConflict::Error,
        index_build_mode: ImportIndexBuildMode::DeferredFullCheckpoint,
    };
    let length = 524_289;
    let chunk = input("P".repeat(length)).chunks.pop().unwrap();
    let mut held = context.workspace_reservation();
    let mut visits = 0;
    let result = db.import_preprocessed_rows(options, context.clone(), |emit| {
        visits += 1;
        emit(
            ImportRow::Document {
                input: ImportDocument {
                    external_id: "document".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                },
                line: 1,
            },
            context.reservation(),
        )?;
        // Apply concurrent scratch pressure after the document has been staged.
        let occupied = context.available_workspace_bytes() - length / 2;
        held.grow_to(occupied, occupied, "operation_scratch")?;
        visits += 1;
        let (result, allocations) = crate::test_alloc::key_allocations(length, || {
            emit(
                ImportRow::Chunk {
                    document_external_id: "document".into(),
                    chunk,
                    line: 1,
                },
                context.reservation(),
            )
        });
        assert_eq!(allocations.allocated, 0);
        result
    });
    let error = result.unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted, "{error}");
    assert_eq!(
        error_detail_value(&error, "operation"),
        Some("operation_scratch")
    );
    let requested = error_detail_value(&error, "requested_bytes")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let limit = error_detail_value(&error, "limit_bytes")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    assert!(requested > limit + length / 4, "{error}");
    assert_eq!(visits, 2);
    held.retain(0);
    assert_eq!(context.available_workspace_bytes(), capacity);
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(db.stats().unwrap().last_commit_id, commit);
    assert_eq!(db.stats().unwrap().document_count, 0);
    assert_eq!(db.temporary_disk_budget().used(), 0);
    assert_eq!(db.temporary_work.used(), 0);
    collection.insert_document(input("PROJECT".into())).unwrap();
    assert_eq!(
        collection
            .get_chunk_by_external_id("chunk")
            .unwrap()
            .chunk_id,
        "chk_0000000000000001"
    );
    drop((collection, db));
    fs::remove_file(path).unwrap();
}
