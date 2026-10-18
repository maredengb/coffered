use super::*;

fn fixture(label: &str) -> (PathBuf, Database, CollectionHandle) {
    let directory = temp_path(label);
    fs::create_dir(&directory).unwrap();
    let db = Database::create(
        directory.join("data.coffer"),
        DatabaseConfig {
            cache_budget_mb: Some(8),
            ..DatabaseConfig::default()
        },
    )
    .unwrap();
    let col = db
        .create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    col.insert_document(DocumentInput {
        external_id: "d".into(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: vec![ChunkInput {
            chunk_external_id: "c".into(),
            ordinal: 0,
            text: "needle".into(),
            metadata: JsonValue::object(),
            source_span: None,
        }],
    })
    .unwrap();
    (directory, db, col)
}

fn write_inputs(directory: &Path, documents: &str, chunks: &str, embeddings: &str) -> PathBuf {
    let with_metadata = |rows: &str| {
        rows.lines()
            .map(|row| {
                let mut value = parse(row).unwrap();
                value
                    .as_object_mut()
                    .unwrap()
                    .insert("metadata".into(), JsonValue::object());
                value.to_compact_string() + "\n"
            })
            .collect::<String>()
    };
    fs::write(directory.join("documents.jsonl"), with_metadata(documents)).unwrap();
    fs::write(directory.join("chunks.jsonl"), with_metadata(chunks)).unwrap();
    fs::write(directory.join("embeddings.jsonl"), embeddings).unwrap();
    let path = directory.join("manifest.json");
    fs::write(&path, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl"}"#).unwrap();
    path
}

fn assert_input_duplicate(error: Error, source: &Path, line: usize) {
    assert_eq!(error.code(), ErrorCode::DuplicateExternalId, "{error:?}");
    let detail = error.to_json();
    let detail = detail.get("detail").unwrap();
    assert_eq!(
        detail.get("path").and_then(JsonValue::as_str),
        source.to_str()
    );
    assert_eq!(
        detail.get("first_line").and_then(JsonValue::as_str),
        Some("1")
    );
    assert_eq!(
        detail.get("line").and_then(JsonValue::as_str),
        Some(line.to_string().as_str())
    );
}

#[test]
fn manifest_identity_duplicates_precede_missing_parents_and_target_conflicts() {
    let (directory, db, col) = fixture("import-priority-identities");
    let before = fs::read(directory.join("data.coffer")).unwrap();
    for (source, documents, chunks) in [
        ("documents.jsonl", "{\"external_id\":\"d\"}\n{\"external_id\":\"d\"}\n", ""),
        ("chunks.jsonl", "{\"external_id\":\"new\"}\n", "{\"document_external_id\":\"missing\",\"chunk_external_id\":\"duplicate\",\"ordinal\":0,\"text\":\"x\"}\n{\"document_external_id\":\"missing\",\"chunk_external_id\":\"duplicate\",\"ordinal\":1,\"text\":\"x\"}\n"),
        ("chunks.jsonl", "{\"external_id\":\"new\"}\n", "{\"document_external_id\":\"new\",\"chunk_external_id\":\"c\",\"ordinal\":0,\"text\":\"x\"}\n{\"document_external_id\":\"new\",\"chunk_external_id\":\"c\",\"ordinal\":1,\"text\":\"x\"}\n"),
    ] {
        let manifest = write_inputs(&directory, documents, chunks, "");
        for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
            assert_input_duplicate(
                db.import_manifest(&manifest, conflict).unwrap_err(),
                &directory.join(source),
                2,
            );
            assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
            assert_eq!(db.temporary_disk_budget().used(), 0);
        }
    }
    drop((col, db));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn target_configuration_refusal_does_not_hide_any_identity_domain_duplicates() {
    let (directory, db, col) = fixture("import-priority-config");
    let before = fs::read(directory.join("data.coffer")).unwrap();
    for (source, documents, chunks, embeddings) in [
        ("documents.jsonl", "{\"external_id\":\"new\"}\n{\"external_id\":\"new\"}\n", "", ""),
        ("chunks.jsonl", "{\"external_id\":\"new\"}\n", "{\"document_external_id\":\"new\",\"chunk_external_id\":\"next\",\"ordinal\":0,\"text\":\"x\"}\n{\"document_external_id\":\"new\",\"chunk_external_id\":\"next\",\"ordinal\":1,\"text\":\"x\"}\n", ""),
        ("embeddings.jsonl", "{\"external_id\":\"new\"}\n", "{\"document_external_id\":\"new\",\"chunk_external_id\":\"next\",\"ordinal\":0,\"text\":\"x\"}\n", "{\"chunk_external_id\":\"next\",\"model_name\":\"M\",\"dim\":3,\"vector\":[1,0,0]}\n{\"chunk_external_id\":\"next\",\"model_name\":\"M\",\"dim\":3,\"vector\":[1,0,0]}\n"),
    ] {
        for configuration in ["dimension", "metric", "inherited_dimension"] {
            let manifest = write_inputs(&directory, documents, chunks, embeddings);
            let mut value = parse(&fs::read_to_string(&manifest).unwrap()).unwrap();
            let object = value.as_object_mut().unwrap();
            match configuration {
                "dimension" => { object.insert("embedding_dim".into(), JsonValue::Number(3.0)); }
                "metric" => { object.insert("distance_metric".into(), JsonValue::String("dot".into())); }
                _ => { object.remove("embedding_dim"); }
            }
            fs::write(&manifest, value.to_compact_string()).unwrap();
            for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
                assert_input_duplicate(
                    db.import_manifest(&manifest, conflict).unwrap_err(),
                    &directory.join(source),
                    2,
                );
                assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
                assert_eq!(db.temporary_disk_budget().used(), 0);
            }
        }
    }
    drop((col, db));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn input_duplicate_priority_covers_embedding_api_and_jsonl_target_dimensions() {
    let (directory, db, col) = fixture("import-priority-embedding-dimension");
    let before = fs::read(directory.join("data.coffer")).unwrap();
    let row = || EmbeddingInput {
        chunk_external_id: "c".into(),
        model_name: "M".into(),
        dim: 3,
        vector: vec![1.0, 0.0, 0.0],
    };
    let input = directory.join("embeddings.jsonl");
    fs::write(&input, "{\"chunk_external_id\":\"c\",\"model_name\":\"M\",\"dim\":3,\"vector\":[1,0,0]}\n{\"chunk_external_id\":\"c\",\"model_name\":\"M\",\"dim\":3,\"vector\":[1,0,0]}\n").unwrap();
    for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
        assert_eq!(
            db.put_embeddings_in_collection(col.collection_id(), vec![row(), row()], conflict)
                .unwrap_err()
                .code(),
            ErrorCode::DuplicateExternalId
        );
        assert_input_duplicate(
            db.import_embeddings_jsonl(col.collection_id(), &input, conflict)
                .unwrap_err(),
            &input,
            2,
        );
        assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
        assert_eq!(db.temporary_disk_budget().used(), 0);
    }
    assert_eq!(
        col.put_embeddings(vec![row()]).unwrap_err().code(),
        ErrorCode::InvalidEmbeddingDimension
    );
    drop((col, db));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn spilled_duplicate_keys_and_deleted_name_refusals_keep_their_real_error() {
    let (directory, db, col) = fixture("import-priority-spill");
    db.drop_collection("notes").unwrap();
    let before = fs::read(directory.join("data.coffer")).unwrap();
    let first = "{\"external_id\":\"d\"}\n";
    let prefix = "p".repeat(256);
    let middle = (0..250)
        .map(|id| format!("{{\"external_id\":\"{prefix}{id}\"}}\n"))
        .collect::<String>();
    let manifest = write_inputs(&directory, &(first.to_owned() + &middle + first), "", "");
    for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
        assert_input_duplicate(
            db.import_manifest(&manifest, conflict).unwrap_err(),
            &directory.join("documents.jsonl"),
            252,
        );
        assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
        assert_eq!(db.temporary_disk_budget().used(), 0);
    }
    write_inputs(&directory, first, "", "");
    assert_eq!(
        db.import_manifest(&manifest, OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::CollectionDeletedNameReserved
    );
    assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
    drop((col, db));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn ordinal_spill_duplicates_do_not_mask_identity_duplicates_or_reuse_failed_sorters() {
    let (directory, db, col) = fixture("import-priority-ordinal-spill");
    let before = fs::read(directory.join("data.coffer")).unwrap();
    let document_id = "p".repeat(512);
    let documents = format!("{{\"external_id\":\"{document_id}\"}}\n");
    let chunk = |id: &str, ordinal| {
        format!("{{\"document_external_id\":\"{document_id}\",\"chunk_external_id\":\"{id}\",\"ordinal\":{ordinal},\"text\":\"needle\"}}\n")
    };
    for identity_duplicate in [true, false] {
        let mut chunks = chunk("a", 0) + &chunk(if identity_duplicate { "a" } else { "b" }, 0);
        for id in 0..250 {
            chunks.push_str(&chunk(&format!("c{id}"), id + 1));
        }
        let manifest = write_inputs(&directory, &documents, &chunks, "");
        let error = db
            .import_manifest(&manifest, OnConflict::Replace)
            .unwrap_err();
        if identity_duplicate {
            assert_input_duplicate(error, &directory.join("chunks.jsonl"), 2);
        } else {
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            assert!(error.message().contains("ordinal must be unique"));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "first_line" && value == "1"));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "line" && value == "2"));
        }
        assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
        assert_eq!(db.temporary_disk_budget().used(), 0);
    }
    drop((col, db));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn embedding_identity_push_failure_is_not_deferred_as_a_target_conflict() {
    let (directory, db, col) = fixture("import-priority-embedding-spill");
    let before = fs::read(directory.join("data.coffer")).unwrap();
    let input = directory.join("embeddings.jsonl");
    let row = |id: &str| {
        format!(
            "{{\"chunk_external_id\":\"{id}\",\"model_name\":\"M\",\"dim\":2,\"vector\":[1,0]}}\n"
        )
    };
    let mut rows = row("c") + &row("c");
    let prefix = "p".repeat(256);
    for id in 0..1001 {
        rows.push_str(&row(&format!("{prefix}{id}")));
    }
    fs::write(&input, rows).unwrap();
    for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
        assert_input_duplicate(
            db.import_embeddings_jsonl(col.collection_id(), &input, conflict)
                .unwrap_err(),
            &input,
            2,
        );
        assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
        assert_eq!(db.temporary_disk_budget().used(), 0);
    }
    drop((col, db));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn unique_inputs_preserve_target_refusals_and_independent_identity_domains() {
    let (directory, db, col) = fixture("import-priority-valid");
    let before = fs::read(directory.join("data.coffer")).unwrap();
    let manifest = write_inputs(
        &directory,
        "{\"external_id\":\"new\"}\n",
        "{\"document_external_id\":\"missing\",\"chunk_external_id\":\"next\",\"ordinal\":0,\"text\":\"needle\"}\n",
        "",
    );
    let error = db.import_manifest(&manifest, OnConflict::Skip).unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
    assert!(error.message().contains("missing document_external_id"));
    assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
    let manifest = write_inputs(&directory, "{\"external_id\":\"new\"}\n", "", "");
    let mut value = parse(&fs::read_to_string(&manifest).unwrap()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("embedding_dim".into(), JsonValue::Number(3.0));
    fs::write(&manifest, value.to_compact_string()).unwrap();
    assert_eq!(
        db.import_manifest(&manifest, OnConflict::Skip)
            .unwrap_err()
            .code(),
        ErrorCode::InvalidEmbeddingDimension
    );
    assert_eq!(fs::read(directory.join("data.coffer")).unwrap(), before);
    let invalid = || EmbeddingInput {
        chunk_external_id: "c".into(),
        model_name: "M".into(),
        dim: 3,
        vector: vec![1.0, 0.0],
    };
    let error = col.put_embeddings(vec![invalid(), invalid()]).unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidEmbeddingDimension);
    assert!(error.message().contains("vector length"));
    let manifest = write_inputs(
        &directory,
        "{\"external_id\":\"same\"}\n",
        "{\"document_external_id\":\"same\",\"chunk_external_id\":\"same\",\"ordinal\":0,\"text\":\"needle\"}\n",
        "{\"chunk_external_id\":\"same\",\"model_name\":\"M\",\"dim\":2,\"vector\":[1,0]}\n",
    );
    let summary = db.import_manifest(&manifest, OnConflict::Error).unwrap();
    assert_eq!(
        (
            summary.documents_inserted,
            summary.chunks_inserted,
            summary.embeddings_inserted
        ),
        (1, 1, 1)
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
    assert_eq!(db.temporary_disk_budget().used(), 0);
    drop((col, db));
    let reopened = Database::open(directory.join("data.coffer")).unwrap();
    assert_eq!(
        reopened
            .collection("notes")
            .unwrap()
            .stats()
            .unwrap()
            .embedding_count,
        1
    );
    drop(reopened);
    fs::remove_dir_all(directory).unwrap();
}
