use super::*;
use crate::import::warnings::ImportWarnings;

#[test]
fn warning_refusal_stops_before_later_rows_and_preserves_the_committed_file() {
    let path = cached_vector_batch_fixture("import-warning-refusal");
    let db = Database::open(&path).unwrap();
    let before = fs::read(&path).unwrap();
    let context = db.json_context();
    let available = context.available_workspace_bytes();
    let mut warnings = ImportWarnings::new(&context).unwrap();
    let mut held = context.workspace_reservation();
    let mut visits = 0;
    let result = db.import_preprocessed_rows(
        ManifestImportOptions {
            tokenizer: Default::default(),
            collection_name: "imported".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: None,
            document_path: "documents.jsonl".into(),
            chunk_path: "chunks.jsonl".into(),
            embedding_path: None,
            on_conflict: OnConflict::Error,
            index_build_mode: ImportIndexBuildMode::DeferredFullCheckpoint,
        },
        context.clone(),
        &mut warnings,
        |emit| {
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
                context.reservation().into(),
            )?;
            for ordinal in 0..3 {
                if ordinal == 1 {
                    let payload: usize = ImportWarnings::SOURCE_SPAN_CONFLICT
                        .iter()
                        .map(|s| s.len())
                        .sum();
                    let occupied = context.available_workspace_bytes() - (payload - 1);
                    held.grow_to(occupied, occupied, "operation_scratch")?;
                }
                visits += 1;
                emit(
                    ImportRow::Chunk {
                        document_external_id: "document".into(),
                        chunk: ChunkInput {
                            chunk_external_id: format!("chunk-{ordinal}"),
                            text: "needle".into(),
                            ordinal,
                            metadata: JsonValue::object(),
                            source_span: None,
                        },
                        line: ordinal + 1,
                        source_span_metadata_conflict: true,
                    },
                    context.reservation().into(),
                )?;
            }
            Ok(())
        },
    );
    let error = result.unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
    assert_eq!(
        error_detail_value(&error, "operation"),
        Some("operation_scratch")
    );
    assert_eq!(visits, 3);
    assert_eq!(fs::read(&path).unwrap(), before);
    let (values, work) = warnings.into_parts();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].code, "source_span_metadata_conflict");
    drop(values);
    drop(work);
    drop(held);
    assert_eq!(context.available_workspace_bytes(), available);
    assert_eq!(db.temporary_disk_budget().used(), 0);
    assert!(db.collection("imported").is_err());
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn locked_warning_admission_reclaims_cache_but_never_retries_native_failures() {
    for native in [false, true] {
        let path = cached_vector_batch_fixture(&format!("import-warning-reclaim-{native}"));
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let cached_bytes = {
            let mut inner = db.lock().unwrap();
            let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
            inner.persistent_vector_segments = Some(cached);
            inner.optional_cache_bytes()
        };
        assert!(cached_bytes > 0);
        let before = fs::read(&path).unwrap();
        let mut warnings = ImportWarnings::new(&context).unwrap();
        let mut visits = 0;
        let result = db.import_preprocessed_rows(
            ManifestImportOptions {
                tokenizer: Default::default(),
                collection_name: "imported".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: None,
                document_path: "documents.jsonl".into(),
                chunk_path: "chunks.jsonl".into(),
                embedding_path: None,
                on_conflict: OnConflict::Error,
                index_build_mode: ImportIndexBuildMode::DeferredFullCheckpoint,
            },
            context.clone(),
            &mut warnings,
            |emit| {
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
                    context.reservation().into(),
                )?;
                let row = ImportRow::Chunk {
                    document_external_id: "document".into(),
                    chunk: ChunkInput {
                        chunk_external_id: "chunk".into(),
                        text: "needle".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    },
                    line: 1,
                    source_span_metadata_conflict: true,
                };
                visits += 1;
                if native {
                    crate::test_alloc::fail_byte_allocation_once(
                        "source_span_metadata_conflict".len(),
                        || emit(row, context.reservation().into()),
                    )
                } else {
                    let payload: usize = ImportWarnings::SOURCE_SPAN_CONFLICT
                        .iter()
                        .map(|s| s.len())
                        .sum();
                    let blocker = context
                        .cache_reservation(
                            (context.available_total_bytes() - (payload - 1)) as u128,
                            usize::MAX,
                        )
                        .unwrap();
                    assert_eq!(context.available_workspace_bytes(), payload - 1);
                    let result = emit(row, context.reservation().into());
                    drop(blocker);
                    result
                }
            },
        );
        assert_eq!(visits, 2);
        let (values, work) = warnings.into_parts();
        if native {
            assert_eq!(result.unwrap_err().code(), ErrorCode::IoError);
            assert!(values.is_empty());
            assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached_bytes);
            assert_eq!(fs::read(&path).unwrap(), before);
            assert!(db.collection("imported").is_err());
        } else {
            // The warning now fits after cache reclamation; the deliberately
            // retained pressure may still refuse later Catalog preparation.
            let error = result.unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("catalog_build")
            );
            assert_eq!(values.len(), 1);
            assert_eq!(db.lock().unwrap().optional_cache_bytes(), 0);
            assert_eq!(fs::read(&path).unwrap(), before);
            assert!(db.collection("imported").is_err());
        }
        drop(values);
        drop(work);
        assert_eq!(db.temporary_disk_budget().used(), 0);
        drop(db);
        assert_eq!(
            context.available_workspace_bytes(),
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
        fs::remove_file(path).unwrap();
    }
}
