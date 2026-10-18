use super::*;

fn fixture(label: &str) -> (PathBuf, Database, CollectionHandle) {
    let path = temp_path(label);
    let db = Database::create(
        &path,
        DatabaseConfig {
            durability: Durability::Fast,
            ..DatabaseConfig::default()
        },
    )
    .unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            name: "close".into(),
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
            title: Some("Title".into()),
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "chunk".into(),
                text: "closing keeps committed data".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
    (path, db, collection)
}

fn assert_closed<T>(result: Result<T>) {
    let error = match result {
        Ok(_) => panic!("closed database operation succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(error.message(), "database handle is closed");
    assert!(error
        .detail()
        .iter()
        .any(|(key, value)| { key == "reason" && value == "database_closed" }));
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::NotConnected
    );
}

#[test]
fn explicit_close_invalidates_aliases_but_preserves_owned_results_and_independent_readers() {
    let (path, db, collection) = fixture("close-shared-runtime");
    let alias = db.clone();
    let collection_alias = collection.clone();
    let query = collection.search().text("committed");
    let results = query.clone().execute().unwrap();
    let document = collection.get_document_by_external_id("document").unwrap();
    let reader = Database::open_read_only(&path).unwrap();
    let readonly_alias = reader.clone();
    let reader_collection = reader.collection("close").unwrap();
    let before = fs::read(&path).unwrap();

    db.close().unwrap();
    assert_closed(alias.stats());
    assert_closed(alias.index_stats());
    assert_closed(alias.collection("close"));
    assert_closed(alias.list_collections());
    assert_closed(alias.check(CheckOptions {
        verify_indexes: true,
    }));
    assert_closed(alias.compact(CompactOptions {
        rebuild_indexes: false,
    }));
    assert_closed(alias.rebuild_index(None));
    assert_closed(alias.drop_collection("close"));
    assert_closed(collection.stats());
    assert_closed(collection.get_document_by_external_id("document"));
    assert_closed(collection_alias.get_chunk_by_external_id("chunk"));
    assert_closed(query.execute());
    assert_closed(collection.put_embeddings(vec![EmbeddingInput {
        chunk_external_id: "chunk".into(),
        model_name: "m".into(),
        dim: 2,
        vector: vec![1.0, 0.0],
    }]));
    assert_eq!(collection.name(), "close");
    assert_eq!(results.results.len(), 1);
    assert_eq!(document.external_id, "document");
    assert_eq!(reader_collection.stats().unwrap().chunk_count, 1);
    assert_eq!(
        reader_collection
            .search()
            .text("committed")
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    let reopened = Database::open(&path).unwrap();
    assert_eq!(
        reopened
            .collection("close")
            .unwrap()
            .stats()
            .unwrap()
            .chunk_count,
        1
    );
    reopened.close().unwrap();
    alias.clone().close().unwrap();
    alias.close().unwrap();
    reader
        .close_with(|storage| storage.flush_committed_with(|_| panic!("reader synced a file")))
        .unwrap();
    readonly_alias.close().unwrap();
    assert_closed(reader_collection.stats());
    fs::remove_file(path).unwrap();
}

#[test]
fn closing_one_clone_does_not_discard_still_owned_output_credit() {
    let (path, db, collection) = fixture("close-owned-output");
    let context = db.json_context();
    let (results, work) = db
        .search_in_collection(
            collection.collection_id(),
            Some("committed"),
            None,
            &JsonValue::object(),
            None,
            false,
            None,
            10,
            None,
            Some(context.clone()),
        )
        .unwrap();
    let retained = work.json_bytes() + work.workspace_bytes();
    assert!(retained > 0);
    db.close().unwrap();
    assert_eq!(
        context.available_workspace_bytes(),
        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - retained
    );
    assert_eq!(results.results[0].chunk_external_id, "chunk");
    drop((results, work));
    assert_eq!(
        context.available_workspace_bytes(),
        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    fs::remove_file(path).unwrap();
}

#[cfg(unix)]
#[test]
fn close_propagates_real_sync_error_but_releases_runtime_and_never_retries_it() {
    let (path, db, collection) = fixture("close-sync-failure");
    let alias = db.clone();
    let before = fs::read(&path).unwrap();
    let calls = Cell::new(0);
    let error = db
        .close_with(|storage| {
            storage.flush_committed_with(|_| {
                calls.set(calls.get() + 1);
                // Inject an actual unsupported sync error at the writer's close IO boundary.
                File::open("/dev/null")?.sync_data()
            })
        })
        .unwrap_err();
    assert_eq!(calls.get(), 1);
    assert_eq!(error.code(), ErrorCode::IoError);
    assert!(std::error::Error::source(&error).is_some());
    assert_closed(collection.stats());
    alias
        .close_with(|_| panic!("already closed runtime retried flushing"))
        .unwrap();
    let reopened = Database::open(&path).unwrap();
    assert_eq!(
        reopened
            .collection("close")
            .unwrap()
            .stats()
            .unwrap()
            .chunk_count,
        1
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    reopened.close().unwrap();
    fs::remove_file(path).unwrap();
}

#[test]
fn fast_writer_close_syncs_once_and_repeated_close_does_not_sync() {
    let (path, db, collection) = fixture("close-fast-sync");
    let alias = db.clone();
    let calls = Cell::new(0);
    db.close_with(|storage| {
        storage.flush_committed_with(|file| {
            calls.set(calls.get() + 1);
            file.sync_data()
        })
    })
    .unwrap();
    assert_eq!(calls.get(), 1);
    alias
        .close_with(|_| panic!("closed runtime repeated sync"))
        .unwrap();
    assert_closed(collection.stats());
    Database::open(&path).unwrap().close().unwrap();
    fs::remove_file(path).unwrap();
}

#[test]
fn close_serializes_with_inflight_operations_and_other_close_calls() {
    let (path, db, collection) = fixture("close-inflight");
    let alias = db.clone();
    let second_close = db.clone();
    let guard = db.lock().unwrap();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let first = std::thread::spawn(move || {
        alias.close_with(|storage| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            storage.flush_committed()
        })
    });
    assert!(started_rx.try_recv().is_err());
    assert_eq!(guard.catalog.live_collections().count(), 1);
    drop(guard);
    started_rx.recv().unwrap();
    assert!(matches!(
        db.inner.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let second = std::thread::spawn(move || {
        let result = second_close.close();
        done_tx.send(result).unwrap();
    });
    assert!(done_rx.try_recv().is_err());
    release_tx.send(()).unwrap();
    first.join().unwrap().unwrap();
    done_rx.recv().unwrap().unwrap();
    second.join().unwrap();
    assert_closed(collection.stats());
    Database::open(&path).unwrap().close().unwrap();
    db.close().unwrap();
    fs::remove_file(path).unwrap();
}

#[test]
fn close_cleans_up_poisoned_runtime_without_resuming_business_operations() {
    let (path, db, collection) = fixture("close-poisoned-runtime");
    let alias = db.clone();
    assert!(std::thread::spawn(move || {
        let _guard = alias.lock().unwrap();
        panic!("inject lock poisoning");
    })
    .join()
    .is_err());
    assert_eq!(
        collection.stats().unwrap_err().code(),
        ErrorCode::LockContention
    );
    assert_eq!(db.close().unwrap_err().code(), ErrorCode::LockContention);
    Database::open(&path).unwrap().close().unwrap();
    fs::remove_file(path).unwrap();
}
