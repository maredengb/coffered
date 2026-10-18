use super::*;

fn fixture(
    label: &str,
    explicit: Option<&str>,
    chunks: usize,
) -> (PathBuf, Database, CollectionHandle) {
    normalized_fixture(label, explicit, chunks, false)
}

fn normalized_fixture(
    label: &str,
    explicit: Option<&str>,
    chunks: usize,
    vectors_normalized: bool,
) -> (PathBuf, Database, CollectionHandle) {
    let path = temp_path(label);
    let database = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = database
        .create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: explicit.map(str::to_owned),
            vectors_normalized,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    collection.insert_document(document(chunks)).unwrap();
    (path, database, collection)
}

fn document(chunks: usize) -> DocumentInput {
    DocumentInput {
        external_id: "document".into(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: (0..chunks)
            .map(|ordinal| ChunkInput {
                chunk_external_id: format!("c{ordinal}"),
                ordinal,
                text: "model retrieval".into(),
                metadata: JsonValue::object(),
                source_span: None,
            })
            .collect(),
    }
}

fn embedding(id: usize, model: &str) -> EmbeddingInput {
    EmbeddingInput {
        chunk_external_id: format!("c{id}"),
        model_name: model.into(),
        dim: 2,
        vector: vec![1.0, 0.0],
    }
}

fn put(
    database: &Database,
    collection: &CollectionHandle,
    rows: Vec<EmbeddingInput>,
    conflict: OnConflict,
) -> Result<InsertSummary> {
    database.put_embeddings_in_collection(collection.collection_id(), rows, conflict)
}

#[test]
fn derived_model_binding_skip_and_input_duplicate_priority_are_atomic() {
    let (path, db, col) = fixture("model-derived-binding", None, 2);
    let original = fs::read(&path).unwrap();
    let error = col
        .put_embeddings(vec![embedding(0, "M"), embedding(1, "N")])
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::EmbeddingModelMismatch);
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(col.stats().unwrap().embedding_count, 0);
    col.put_embeddings(vec![embedding(0, "M")]).unwrap();
    let committed = fs::read(&path).unwrap();
    assert_eq!(
        put(&db, &col, vec![embedding(99, "M")], OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::ChunkNotFound
    );
    assert_eq!(fs::read(&path).unwrap(), committed);
    assert_eq!(
        put(&db, &col, vec![embedding(0, "N")], OnConflict::Skip)
            .unwrap()
            .embeddings_skipped,
        1
    );
    assert_eq!(fs::read(&path).unwrap(), committed);
    for rows in [vec![embedding(0, "N")], vec![embedding(1, "m")]] {
        assert_eq!(
            put(&db, &col, rows, OnConflict::Replace)
                .unwrap_err()
                .code(),
            ErrorCode::EmbeddingModelMismatch
        );
        assert_eq!(fs::read(&path).unwrap(), committed);
    }
    for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
        assert_eq!(
            put(
                &db,
                &col,
                vec![embedding(1, "N"), embedding(1, "N")],
                conflict
            )
            .unwrap_err()
            .code(),
            ErrorCode::DuplicateExternalId
        );
        assert_eq!(fs::read(&path).unwrap(), committed);
    }
    let mut malformed = embedding(0, "N");
    malformed.vector[0] = f32::NAN;
    assert_eq!(
        put(&db, &col, vec![malformed], OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::InvalidVectorLength
    );
    assert_eq!(
        col.search()
            .vector(vec![1.0, 0.0])
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    assert_eq!(
        col.search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("N".into()),
                dim: 2,
                vector: vec![1.0, 0.0]
            })
            .execute()
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert!(db
        .lock()
        .unwrap()
        .catalog
        .collection_by_id(col.collection_id())
        .unwrap()
        .embedding_model
        .is_none());
    drop((col, db));
    let reopened = Database::open(&path).unwrap();
    assert_eq!(
        reopened
            .collection("notes")
            .unwrap()
            .put_embeddings(vec![embedding(1, "N")])
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    drop(reopened);
    fs::remove_file(path).unwrap();
}

fn jsonl_row(id: usize, model: &str) -> String {
    format!("{{\"chunk_external_id\":\"c{id}\",\"model_name\":\"{model}\",\"dim\":2,\"vector\":[1,0]}}\n")
}

#[test]
fn embedding_stream_preflight_keeps_one_model_across_batches_and_duplicate_priority() {
    let (path, db, col) = fixture("model-stream-binding", None, 1001);
    let input = temp_path("model-stream-input");
    let original = fs::read(&path).unwrap();
    let prefix = (0..1000).map(|id| jsonl_row(id, "M")).collect::<String>();
    fs::write(&input, prefix.clone() + &jsonl_row(1000, "N")).unwrap();
    assert_eq!(
        db.import_embeddings_jsonl(col.collection_id(), &input, OnConflict::Error)
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
        fs::write(
            &input,
            prefix.clone() + &jsonl_row(1000, "N") + &jsonl_row(1000, "N"),
        )
        .unwrap();
        assert_eq!(
            db.import_embeddings_jsonl(col.collection_id(), &input, conflict)
                .unwrap_err()
                .code(),
            ErrorCode::DuplicateExternalId
        );
        assert_eq!(fs::read(&path).unwrap(), original);
    }
    fs::write(&input, prefix + &jsonl_row(1000, "M")).unwrap();
    assert_eq!(
        db.import_embeddings_jsonl(col.collection_id(), &input, OnConflict::Error)
            .unwrap()
            .embeddings_inserted,
        1001
    );
    let committed = fs::read(&path).unwrap();
    fs::write(&input, jsonl_row(0, "N")).unwrap();
    assert_eq!(
        db.import_embeddings_jsonl(col.collection_id(), &input, OnConflict::Skip)
            .unwrap()
            .embeddings_skipped,
        1
    );
    assert_eq!(fs::read(&path).unwrap(), committed);
    assert_eq!(
        db.import_embeddings_jsonl(col.collection_id(), &input, OnConflict::Replace)
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(fs::read(&path).unwrap(), committed);
    assert_eq!(db.temporary_work.used(), 0);
    drop((col, db));
    fs::remove_file(path).unwrap();
    fs::remove_file(input).unwrap();
}

fn manifest(directory: &Path, declaration: Option<&str>, rows: &str, chunks: usize) -> PathBuf {
    fs::create_dir(directory).unwrap();
    let declared = declaration.map_or(String::new(), |model| {
        format!(",\"embedding_model\":\"{model}\"")
    });
    let path = directory.join("manifest.json");
    fs::write(&path, format!("{{\"schema_version\":\"0.1\",\"collection\":\"notes\",\"embedding_dim\":2,\"distance_metric\":\"cosine\"{declared},\"documents\":\"documents.jsonl\",\"chunks\":\"chunks.jsonl\",\"embeddings\":\"embeddings.jsonl\"}}")).unwrap();
    fs::write(
        directory.join("documents.jsonl"),
        "{\"external_id\":\"document\",\"metadata\":{}}\n",
    )
    .unwrap();
    fs::write(directory.join("chunks.jsonl"), (0..chunks).map(|ordinal| format!("{{\"document_external_id\":\"document\",\"chunk_external_id\":\"c{ordinal}\",\"ordinal\":{ordinal},\"text\":\"model retrieval\",\"metadata\":{{}}}}\n")).collect::<String>()).unwrap();
    fs::write(directory.join("embeddings.jsonl"), rows).unwrap();
    path
}

#[test]
fn manifest_model_binding_precedes_every_commit_and_survives_replace_cascade() {
    for explicit in [None, Some("M")] {
        let (path, db, col) = fixture("model-manifest-cascade", explicit, 2);
        col.put_embeddings(vec![embedding(0, "M")]).unwrap();
        let original = fs::read(&path).unwrap();
        let directory = temp_path("model-manifest-input");
        let input = manifest(
            &directory,
            None,
            &(jsonl_row(0, "N") + &jsonl_row(1, "N")),
            2,
        );
        assert_eq!(
            db.import_manifest(&input, OnConflict::Replace)
                .unwrap_err()
                .code(),
            ErrorCode::EmbeddingModelMismatch
        );
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::write(
            directory.join("embeddings.jsonl"),
            jsonl_row(0, "M") + &jsonl_row(1, "M"),
        )
        .unwrap();
        assert_eq!(
            db.import_manifest(&input, OnConflict::Replace)
                .unwrap()
                .embeddings_inserted,
            2
        );
        assert_eq!(
            db.lock()
                .unwrap()
                .catalog
                .collection_by_id(col.collection_id())
                .unwrap()
                .embedding_model
                .as_deref(),
            explicit
        );
        drop((col, db));
        fs::remove_file(path).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }
    let path = temp_path("model-manifest-new");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let before = fs::read(&path).unwrap();
    let directory = temp_path("model-manifest-new-input");
    let input = manifest(
        &directory,
        None,
        &(jsonl_row(0, "M") + &jsonl_row(1, "N")),
        2,
    );
    assert_eq!(
        db.import_manifest(&input, OnConflict::Error)
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(db.list_collections().unwrap().is_empty());
    drop(db);
    fs::remove_file(path).unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn manifest_declaration_conflicts_do_not_hide_input_duplicates_or_clear_configuration() {
    let (path, db, col) = fixture("model-manifest-declaration", Some("M"), 1);
    col.put_embeddings(vec![embedding(0, "M")]).unwrap();
    let before = fs::read(&path).unwrap();
    let directory = temp_path("model-manifest-declaration-input");
    let input = manifest(&directory, Some("N"), &jsonl_row(0, "N"), 1);
    assert_eq!(
        db.import_manifest(&input, OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::write(
        directory.join("embeddings.jsonl"),
        jsonl_row(0, "N") + &jsonl_row(0, "N"),
    )
    .unwrap();
    assert_eq!(
        db.import_manifest(&input, OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::DuplicateExternalId
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    drop((col, db));
    fs::remove_file(path).unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn healthy_collection_writes_and_scoped_vector_build_remain_available_during_rescue() {
    let (path, db, bad) = fixture("model-healthy-rescue-scope", None, 2);
    append_legacy_models(&db, &bad, &["M", "N"]);
    let good = db
        .create_collection(CollectionConfig {
            name: "healthy".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    good.insert_document(document(1)).unwrap();
    good.put_embeddings(vec![embedding(0, "Q")]).unwrap();
    let before = fs::read(&path).unwrap();
    assert_eq!(
        good.search()
            .vector(vec![1.0, 0.0])
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    let summary = db.rebuild_index(Some("healthy")).unwrap();
    assert_eq!(summary.rebuild_scope, "transient_vector_collection");
    assert!(summary.rebuild_needed);
    assert_eq!(
        (summary.index_record_count, summary.index_extent_count),
        (0, 0)
    );
    assert_eq!(summary.vector_segment_count, 1);
    assert!(summary.last_index_commit_id.is_none());
    assert_eq!(
        db.rebuild_index(Some("notes")).unwrap_err().code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    let report = db.check(CheckOptions::default()).unwrap();
    assert_eq!(report.live_documents, 2);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.code == "embedding_model_mismatch"));
    drop((bad, good, db));
    let reopened = Database::open(&path).unwrap();
    assert_eq!(
        reopened
            .collection("healthy")
            .unwrap()
            .search()
            .vector(vec![1.0, 0.0])
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    let (offset, len) = {
        let inner = reopened.lock().unwrap();
        let healthy = inner.catalog.collection_by_name("healthy").unwrap();
        let embedding = inner
            .catalog
            .live_embeddings()
            .find(|embedding| embedding.collection_id == healthy.collection_id)
            .unwrap();
        let reference = inner
            .catalog
            .record_ref_for_embedding(&embedding.embedding_id)
            .unwrap();
        (reference.payload.offset, reference.payload.len)
    };
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let position = offset + len - 12;
    file.seek(SeekFrom::Start(position)).unwrap();
    let mut byte = [0];
    std::io::Read::read_exact(&mut file, &mut byte).unwrap();
    file.seek(SeekFrom::Start(position)).unwrap();
    file.write_all(&[byte[0] ^ 1]).unwrap();
    drop(file);
    let damaged = fs::read(&path).unwrap();
    assert_eq!(
        reopened.rebuild_index(Some("healthy")).unwrap_err().code(),
        ErrorCode::ChecksumMismatch
    );
    assert_eq!(fs::read(&path).unwrap(), damaged);
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn manifest_declaration_constrains_writes_without_persisting_a_derived_binding() {
    let (path, db, col) = fixture("model-manifest-unconfigured-declaration", None, 1);
    let directory = temp_path("model-manifest-unconfigured-input");
    let input = manifest(&directory, Some("M"), &jsonl_row(0, "M"), 1);
    db.import_manifest(&input, OnConflict::Replace).unwrap();
    assert!(db
        .lock()
        .unwrap()
        .catalog
        .collection_by_id(col.collection_id())
        .unwrap()
        .embedding_model
        .is_none());
    let before = fs::read(&path).unwrap();
    fs::write(directory.join("embeddings.jsonl"), jsonl_row(0, "N")).unwrap();
    assert_eq!(
        db.import_manifest(&input, OnConflict::Replace)
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    drop((col, db));
    fs::remove_file(path).unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn manifest_missing_skipped_parent_reference_does_not_hide_duplicate_embeddings() {
    let (path, db, col) = fixture("model-manifest-missing-skip-reference", None, 0);
    let before = fs::read(&path).unwrap();
    let directory = temp_path("model-manifest-missing-skip-input");
    let input = manifest(
        &directory,
        None,
        &(jsonl_row(0, "M") + &jsonl_row(0, "M")),
        1,
    );
    assert_eq!(
        db.import_manifest(&input, OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::DuplicateExternalId
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::write(directory.join("embeddings.jsonl"), jsonl_row(0, "M")).unwrap();
    assert_eq!(
        db.import_manifest(&input, OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::ChunkNotFound
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::write(directory.join("chunks.jsonl"), "").unwrap();
    fs::write(
        directory.join("embeddings.jsonl"),
        jsonl_row(0, "M") + &jsonl_row(0, "M"),
    )
    .unwrap();
    assert_eq!(
        db.import_manifest(&input, OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::DuplicateExternalId
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(db
        .lock()
        .unwrap()
        .catalog
        .collection_by_id(col.collection_id())
        .unwrap()
        .embedding_model
        .is_none());
    drop((col, db));
    fs::remove_file(path).unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn partially_committed_text_import_does_not_publish_request_only_model_binding() {
    let (path, db, col) = fixture("model-partial-text-import", None, 0);
    let directory = temp_path("model-partial-text-input");
    let rows = (0..1001).map(|id| jsonl_row(id, "M")).collect::<String>();
    let input = manifest(&directory, None, &rows, 1001);
    let before = db.stats().unwrap().last_commit_id;
    db.lock().unwrap().import_fail_after_batches = Some(1);
    let error = db.import_manifest(&input, OnConflict::Replace).unwrap_err();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert!(db.stats().unwrap().last_commit_id > before);
    assert_eq!(col.stats().unwrap().embedding_count, 0);
    assert!(col.get_chunk_by_external_id("c0").is_ok());
    assert_eq!(
        col.search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("N".into()),
                dim: 2,
                vector: vec![1.0, 0.0]
            })
            .execute()
            .unwrap()
            .results
            .len(),
        0
    );
    col.put_embeddings(vec![embedding(0, "N")]).unwrap();
    assert_eq!(
        col.search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("M".into()),
                dim: 2,
                vector: vec![1.0, 0.0]
            })
            .execute()
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(db.temporary_work.used(), 0);
    drop((col, db));
    let reopened = Database::open(&path).unwrap();
    assert_eq!(
        reopened
            .collection("notes")
            .unwrap()
            .stats()
            .unwrap()
            .embedding_count,
        1
    );
    assert!(reopened
        .check(CheckOptions::default())
        .unwrap()
        .issues
        .iter()
        .all(|issue| issue.code != "embedding_model_mismatch"));
    drop(reopened);
    fs::remove_file(path).unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn independent_clear_rebinds_only_unconfigured_writer_snapshot() {
    for explicit in [None, Some("M")] {
        let (path, db, col) = fixture("model-independent-clear", explicit, 1);
        col.put_embeddings(vec![embedding(0, "M")]).unwrap();
        let reader = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                ..DatabaseOpenOptions::default()
            },
        )
        .unwrap();
        let old = reader.collection("notes").unwrap();
        old.search().vector(vec![1.0, 0.0]).execute().unwrap();
        let document_id = col
            .get_document_by_external_id("document")
            .unwrap()
            .document_id;
        col.delete_document(&document_id).unwrap();
        col.insert_document(document(1)).unwrap();
        let result = col.put_embeddings(vec![embedding(0, "N")]);
        if explicit.is_some() {
            assert_eq!(
                result.unwrap_err().code(),
                ErrorCode::EmbeddingModelMismatch
            );
        } else {
            result.unwrap();
            assert_eq!(
                col.search()
                    .query_embedding(QueryEmbeddingInput {
                        model_name: Some("M".into()),
                        dim: 2,
                        vector: vec![1.0, 0.0]
                    })
                    .execute()
                    .unwrap_err()
                    .code(),
                ErrorCode::EmbeddingModelMismatch
            );
        }
        assert_eq!(old.stats().unwrap().embedding_count, 1);
        assert_eq!(
            old.search()
                .vector(vec![1.0, 0.0])
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        assert_eq!(
            old.search()
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("N".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0]
                })
                .execute()
                .unwrap_err()
                .code(),
            ErrorCode::EmbeddingModelMismatch
        );
        drop((old, reader, col, db));
        fs::remove_file(path).unwrap();
    }
}

fn append_legacy_models(database: &Database, collection: &CollectionHandle, names: &[&str]) {
    let mut inner = database.lock().unwrap();
    let rows = names
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let chunk = inner
                .catalog
                .chunk_by_external_id(collection.collection_id(), &format!("c{index}"))
                .unwrap();
            RecordPayload::EmbeddingPut(Embedding {
                embedding_id: inner.catalog.next_id("emb_", index as u64).unwrap(),
                chunk_id: chunk.chunk_id.clone(),
                collection_id: collection.collection_id().into(),
                chunk_external_id: chunk.chunk_external_id.clone(),
                model_name: (*model).into(),
                dim: 2,
                vector: vec![1.0, 0.0],
                created_at: "1970-01-01T00:00:00Z".into(),
            })
        })
        .collect();
    // Write structurally valid old committed data without the new-input admission gate.
    inner.commit_records(rows).unwrap();
}

#[test]
fn historical_conflict_remains_readable_checkable_and_explicitly_cleanable() {
    for (explicit, names, normalized) in [
        (None, vec!["M", "N"], false),
        (Some("M"), vec!["M", "N"], false),
        (None, vec!["M", "N"], true),
        (Some("M"), vec!["M", "N"], true),
    ] {
        let (path, db, col) = normalized_fixture("model-history-rescue", explicit, 2, normalized);
        append_legacy_models(&db, &col, &names);
        drop((col, db));
        let db = Database::open(&path).unwrap();
        let col = db.collection("notes").unwrap();
        assert_eq!(col.stats().unwrap().embedding_count, 2);
        assert_eq!(
            col.search()
                .text("retrieval")
                .execute()
                .unwrap()
                .results
                .len(),
            2
        );
        let before = fs::read(&path).unwrap();
        let report = db
            .check(CheckOptions {
                verify_indexes: true,
            })
            .unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == "embedding_model_mismatch")
            .unwrap();
        assert_eq!(
            (
                &*issue.stream,
                &*issue.visibility_impact,
                issue.repair_available
            ),
            ("critical", "none", false)
        );
        let bad = col.get_chunk_by_external_id("c1").unwrap();
        let offset = {
            let inner = db.lock().unwrap();
            inner.catalog.record_refs[&inner
                .catalog
                .embedding_for_chunk(&bad.chunk_id)
                .unwrap()
                .embedding_id]
                .offset
        };
        assert_eq!(issue.offset, Some(offset));
        let (_, exit) = crate::cli::run_with_exit_code(vec![
            "coffer".into(),
            "check".into(),
            path.display().to_string(),
        ])
        .unwrap();
        assert_eq!(exit, 2);
        for weights in [None, Some(SearchWeights::new(1.0, 0.0, 0.0).unwrap())] {
            let mut query = col.search().text("retrieval").vector(vec![1.0, 0.0]);
            if let Some(weights) = weights {
                query = query.weights(weights);
            }
            let error = query.execute().unwrap_err();
            assert_eq!(error.code(), ErrorCode::EmbeddingModelMismatch);
            assert_eq!(
                error.to_json().get("recoverable"),
                Some(&JsonValue::Bool(true))
            );
        }
        assert_eq!(
            db.compact(CompactOptions::default()).unwrap_err().code(),
            ErrorCode::EmbeddingModelMismatch
        );
        assert_eq!(
            db.rebuild_index(None).unwrap_err().code(),
            ErrorCode::EmbeddingModelMismatch
        );
        assert_eq!(
            put(&db, &col, vec![embedding(0, "M")], OnConflict::Replace)
                .unwrap_err()
                .code(),
            ErrorCode::EmbeddingModelMismatch
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(
            put(&db, &col, vec![embedding(0, "other")], OnConflict::Skip)
                .unwrap()
                .embeddings_skipped,
            1
        );
        let directory = temp_path("model-history-export");
        db.export_collection_jsonl("notes", &directory).unwrap();
        let manifest =
            parse(&fs::read_to_string(directory.join("export_manifest.json")).unwrap()).unwrap();
        assert_eq!(
            manifest.get("embedding_model").and_then(JsonValue::as_str),
            explicit
        );
        assert_eq!(
            manifest.get("vectors_normalized"),
            Some(&JsonValue::Bool(normalized))
        );
        assert_eq!(
            manifest.get("embedding_dim").and_then(JsonValue::as_f64),
            Some(2.0)
        );
        assert_eq!(
            manifest.get("distance_metric").and_then(JsonValue::as_str),
            Some("cosine")
        );
        let exported = fs::read_to_string(directory.join("embeddings.jsonl")).unwrap();
        assert!(exported.contains("\"M\"") && exported.contains("\"N\""));
        let reader = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                ..DatabaseOpenOptions::default()
            },
        )
        .unwrap();
        col.delete_chunk(&bad.chunk_id).unwrap();
        assert_eq!(
            col.search()
                .vector(vec![1.0, 0.0])
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        assert!(db
            .check(CheckOptions::default())
            .unwrap()
            .issues
            .iter()
            .all(|issue| issue.code != "embedding_model_mismatch"));
        assert_eq!(
            reader
                .collection("notes")
                .unwrap()
                .search()
                .vector(vec![1.0, 0.0])
                .execute()
                .unwrap_err()
                .code(),
            ErrorCode::EmbeddingModelMismatch
        );
        db.compact(CompactOptions::default()).unwrap();
        drop((reader, col, db));
        fs::remove_file(path).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }
}
