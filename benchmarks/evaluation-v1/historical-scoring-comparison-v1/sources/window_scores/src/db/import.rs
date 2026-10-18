use super::*;
use crate::json::{JsonOperationContext, JsonReservation};
use crate::model::ChunkInput;
use crate::storage::PreparedRecordsBuilder;
use crate::temporary::{ExternalKeyLookup, ExternalUniqueKeys};

#[derive(Debug, Default)]
pub(crate) struct ImportBatchMetrics {
    pub(crate) max_batch_chunks: usize,
    pub(crate) committed_batches: usize,
    pub(crate) checkpoint_cycles: usize,
    pub(crate) checkpoint_ready_cycles: usize,
    pub(crate) final_checkpoint_ready: bool,
    pub(crate) commit_time: std::time::Duration,
    pub(crate) checkpoint_time: std::time::Duration,
}

pub(crate) struct ManifestImportOptions {
    pub tokenizer: crate::TokenizerConfig,
    pub collection_name: String,
    pub embedding_dim: usize,
    pub distance_metric: DistanceMetric,
    pub embedding_model: Option<String>,
    pub document_path: PathBuf,
    pub chunk_path: PathBuf,
    pub embedding_path: Option<PathBuf>,
    pub on_conflict: OnConflict,
    pub index_build_mode: ImportIndexBuildMode,
}

pub(crate) struct ImportDocument {
    pub external_id: String,
    pub title: Option<String>,
    pub source_uri: Option<String>,
    pub metadata: JsonValue,
}

pub(crate) enum ImportRow {
    Document {
        input: ImportDocument,
        line: usize,
    },
    Chunk {
        document_external_id: String,
        chunk: ChunkInput,
        line: usize,
    },
    Embedding {
        input: EmbeddingInput,
        line: usize,
    },
}

struct ImportReferences {
    sorting: Option<ExternalUniqueKeys>,
    lookup: Option<ExternalKeyLookup>,
    path: PathBuf,
}

impl ImportReferences {
    fn new(
        database: &Database,
        path: &Path,
        workspace: usize,
        context: &JsonOperationContext,
        message: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            sorting: Some(ExternalUniqueKeys::new(
                &std::env::temp_dir(),
                database.temporary_disk_budget(),
                workspace,
                context,
                ErrorCode::DuplicateExternalId,
                message,
            )?),
            lookup: None,
            path: path.to_path_buf(),
        })
    }

    fn insert(&mut self, key: &str, line: usize, planned: u64) -> Result<()> {
        let sorting = self.sorting.as_mut().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "import reference rows must precede their lookup phase",
            )
        })?;
        sorting
            .push_value(&[key.as_bytes()], line as u64, planned)
            .map_err(|error| error.with_detail("path", self.path.display()))
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(sorting) = self.sorting.take() {
            self.lookup = Some(
                sorting
                    .finish_lookup()
                    .map_err(|error| error.with_detail("path", self.path.display()))?,
            );
        }
        if self.lookup.is_none() {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "cannot reuse failed import reference lookup",
            ));
        }
        Ok(())
    }

    fn get(&mut self, key: &str) -> Result<Option<(u64, u64)>> {
        self.finish()?;
        self.lookup
            .as_mut()
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "cannot reuse failed import reference lookup",
                )
            })?
            .get(key.as_bytes())
            .map_err(|error| error.with_detail("path", self.path.display()))
    }
}

impl Database {
    pub(crate) fn import_preprocessed_rows(
        &self,
        options: ManifestImportOptions,
        context: JsonOperationContext,
        visit: impl FnOnce(&mut dyn FnMut(ImportRow, JsonReservation) -> Result<()>) -> Result<()>,
    ) -> Result<(InsertSummary, ImportBatchMetrics)> {
        let mut work = context.reservation();
        let mut inner = self.lock()?;
        let collection = import_collection(&inner.catalog, &options, &context, &mut work)?;
        let disk = self.temporary_disk_budget();
        let mut records = {
            let Inner {
                text_index,
                persistent_vector_segments,
                lazy_vector_segment,
                ..
            } = &mut *inner;
            PreparedRecordsBuilder::new_in_reclaiming(
                &std::env::temp_dir(),
                disk.limit_bytes(),
                context.clone(),
                disk,
                || {
                    Inner::release_optional_caches(
                        text_index,
                        persistent_vector_segments,
                        lazy_vector_segment,
                        std::iter::empty(),
                    )
                },
            )?
        };
        if !inner
            .catalog
            .collections
            .contains_key(&collection.collection_id)
        {
            records.push(RecordPayload::CollectionCreate(collection.clone()))?;
        }

        let now = time::now_rfc3339();
        let mut summary = InsertSummary::default();
        let mut ordinal_keys: Option<ExternalUniqueKeys> = None;
        let mut embedding_keys: Option<ExternalUniqueKeys> = None;
        let reference_workspace = (inner.active_cache_budget_mb().saturating_mul(1_048_576) / 128)
            .clamp(64 * 1024, 1_048_576);
        let mut documents = ImportReferences::new(
            self,
            &options.document_path,
            reference_workspace,
            &context,
            "document external_id appears more than once in the same import batch",
        )?;
        let mut chunks = ImportReferences::new(
            self,
            &options.chunk_path,
            reference_workspace,
            &context,
            "chunk_external_id appears more than once in the same import batch",
        )?;
        visit(&mut |row, mut row_work| {
            match row {
                ImportRow::Document { input: row, line } => {
                    let input = DocumentInput {
                        external_id: row.external_id,
                        title: row.title,
                        source_uri: row.source_uri,
                        metadata: row.metadata,
                        chunks: Vec::new(),
                    };
                    input.validate_preprocessed(&context)?;
                    let existing = inner
                        .catalog
                        .document_external_ids
                        .get(&collection.collection_id, &input.external_id)
                        .filter(|id| inner.catalog.is_document_live(id));
                    if let Some(id) = existing {
                        match options.on_conflict {
                            OnConflict::Error => {
                                return Err(existing_id("document external_id", &input.external_id))
                            }
                            OnConflict::Skip => {
                                summary.documents_skipped += 1;
                                documents.insert(&input.external_id, line, 0)?;
                                return Ok(());
                            }
                            OnConflict::Replace => {
                                records.push(replacement("document", id, &now))?;
                                summary.documents_replaced += 1;
                            }
                        }
                    }
                    let document_id = inner
                        .catalog
                        .next_id("doc_", summary.documents_inserted as u64)?;
                    documents.insert(
                        &input.external_id,
                        line,
                        summary.documents_inserted as u64 + 1,
                    )?;
                    records.push_retained(
                        RecordPayload::DocumentPut(Document {
                            document_id,
                            collection_id: collection.collection_id.clone(),
                            external_id: input.external_id,
                            title: input.title,
                            source_uri: input.source_uri,
                            metadata: input.metadata,
                            created_at: now.clone(),
                            updated_at: now.clone(),
                        }),
                        &mut row_work,
                    )?;
                    summary.documents_inserted += 1;
                }
                ImportRow::Chunk {
                    document_external_id,
                    chunk,
                    line,
                } => {
                    chunk.validate_preprocessed()?;
                    let (document_line, planned_document) = documents.get(&document_external_id)?.ok_or_else(|| {
                        Error::new(ErrorCode::InvalidJsonlSchema,
                            format!("chunks reference missing document_external_id {document_external_id}"))
                    })?;
                    if ordinal_keys.is_none() {
                        ordinal_keys = Some(ExternalUniqueKeys::new(
                            &std::env::temp_dir(),
                            self.temporary_disk_budget(),
                            reference_workspace,
                            &context,
                            ErrorCode::InvalidJsonlSchema,
                            "chunk ordinal must be unique within a document",
                        )?);
                    }
                    ordinal_keys.as_mut().unwrap().push(
                        &[
                            &document_line.to_le_bytes(),
                            &(chunk.ordinal as u64).to_le_bytes(),
                        ],
                        line as u64,
                    )?;
                    if planned_document == 0 {
                        chunks.insert(&chunk.chunk_external_id, line, 0)?;
                        summary.chunks_skipped += 1;
                        return Ok(());
                    }
                    let document_id = inner.catalog.next_id("doc_", planned_document - 1)?;
                    let existing = inner
                        .catalog
                        .chunk_external_ids
                        .get(&collection.collection_id, &chunk.chunk_external_id)
                        .filter(|id| inner.catalog.is_chunk_live(id));
                    if let Some(id) = existing {
                        match options.on_conflict {
                            OnConflict::Error => {
                                return Err(existing_id(
                                    "chunk_external_id",
                                    &chunk.chunk_external_id,
                                ))
                            }
                            OnConflict::Skip => {
                                chunks.insert(&chunk.chunk_external_id, line, 0)?;
                                summary.chunks_skipped += 1;
                                return Ok(());
                            }
                            OnConflict::Replace => {
                                records.push(replacement("chunk", id, &now))?;
                                summary.chunks_replaced += 1;
                            }
                        }
                    }
                    let chunk_id = inner
                        .catalog
                        .next_id("chk_", summary.chunks_inserted as u64)?;
                    let source_span = chunk.resolved_source_span()?;
                    let token_count =
                        inner.read_reclaiming(|_| token_count_hint_in(&chunk.text, &context))?;
                    chunks.insert(
                        &chunk.chunk_external_id,
                        line,
                        summary.chunks_inserted as u64 + 1,
                    )?;
                    records.push_retained(
                        RecordPayload::ChunkPut(Chunk {
                            chunk_id,
                            document_id,
                            collection_id: collection.collection_id.clone(),
                            chunk_external_id: chunk.chunk_external_id,
                            text_hash: chunk_text_hash(&chunk.text),
                            token_count,
                            text: chunk.text,
                            ordinal: chunk.ordinal,
                            metadata: chunk.metadata,
                            embedding_id: None,
                            source_span,
                            created_at: now.clone(),
                        }),
                        &mut row_work,
                    )?;
                    summary.chunks_inserted += 1;
                }
                ImportRow::Embedding { input, line } => {
                    documents.finish()?;
                    chunks.finish()?;
                    finish_reference_keys(&mut ordinal_keys, Some(&options.chunk_path))?;
                    if embedding_keys.is_none() {
                        embedding_keys = Some(ExternalUniqueKeys::new(
                            &std::env::temp_dir(), self.temporary_disk_budget(), reference_workspace,
                            &context,
                            ErrorCode::DuplicateExternalId,
                            "embedding chunk_external_id appears more than once in the same import batch",
                        )?);
                    }
                    embedding_keys
                        .as_mut()
                        .unwrap()
                        .push(&[input.chunk_external_id.as_bytes()], line as u64)?;
                    input.validate(&collection)?;
                    let (_, planned) = chunks.get(&input.chunk_external_id)?.ok_or_else(|| {
                        Error::new(ErrorCode::InvalidJsonlSchema,
                            format!("embeddings reference chunk_external_id {} not present in this manifest; use import-embeddings for embedding-only updates", input.chunk_external_id))
                    })?;
                    let chunk_id = match planned {
                        0 => match inner.catalog.chunk_by_external_id(
                            &collection.collection_id,
                            &input.chunk_external_id,
                        ) {
                            Ok(chunk) => chunk.chunk_id.clone(),
                            Err(error)
                                if options.on_conflict == OnConflict::Skip
                                    && error.code() == ErrorCode::ChunkNotFound =>
                            {
                                summary.embeddings_skipped += 1;
                                return Ok(());
                            }
                            Err(error) => return Err(error),
                        },
                        number => inner.catalog.next_id("chk_", number - 1)?,
                    };
                    if inner.catalog.embedding_for_chunk(&chunk_id).is_some() {
                        match options.on_conflict {
                            OnConflict::Error => {
                                return Err(existing_id(
                                    "embedding for chunk",
                                    &input.chunk_external_id,
                                ))
                            }
                            OnConflict::Skip => {
                                summary.embeddings_skipped += 1;
                                return Ok(());
                            }
                            OnConflict::Replace => summary.embeddings_replaced += 1,
                        }
                    }
                    records.push(RecordPayload::EmbeddingPut(Embedding {
                        embedding_id: inner
                            .catalog
                            .next_id("emb_", summary.embeddings_inserted as u64)?,
                        chunk_id,
                        collection_id: collection.collection_id.clone(),
                        chunk_external_id: input.chunk_external_id,
                        model_name: input.model_name,
                        dim: input.dim,
                        vector: input.vector,
                        created_at: now.clone(),
                    }))?;
                    summary.embeddings_inserted += 1;
                }
            }
            Ok(())
        })?;
        // All validation and conflict planning uses this same locked target snapshot.
        documents.finish()?;
        chunks.finish()?;
        drop(documents);
        drop(chunks);
        finish_reference_keys(&mut ordinal_keys, Some(&options.chunk_path))?;
        finish_reference_keys(&mut embedding_keys, options.embedding_path.as_deref())?;
        let batches =
            inner.append_import_batches(records.finish()?, options.index_build_mode, &context)?;
        Ok((summary, batches))
    }
}

fn finish_reference_keys(keys: &mut Option<ExternalUniqueKeys>, path: Option<&Path>) -> Result<()> {
    if let Some(keys) = keys.take() {
        keys.finish().map_err(|error| match path {
            Some(path) => error.with_detail("path", path.display()),
            None => error,
        })?;
    }
    Ok(())
}

fn import_collection(
    catalog: &Catalog,
    options: &ManifestImportOptions,
    context: &JsonOperationContext,
    work: &mut JsonReservation,
) -> Result<CollectionInfo> {
    match catalog.collection_by_name(&options.collection_name) {
        Ok(collection) => {
            if collection.tokenizer != options.tokenizer {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "manifest tokenizer does not match collection",
                )
                .with_detail("field", "tokenizer"));
            }
            if collection.embedding_dim != options.embedding_dim {
                return Err(Error::new(
                    ErrorCode::InvalidEmbeddingDimension,
                    "manifest embedding_dim does not match collection",
                )
                .with_detail("expected", collection.embedding_dim)
                .with_detail("actual", options.embedding_dim));
            }
            if collection.distance_metric != options.distance_metric {
                return Err(Error::new(
                    ErrorCode::UnsupportedMetric,
                    "manifest distance_metric does not match collection",
                ));
            }
            if collection.embedding_model != options.embedding_model {
                return Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "manifest embedding_model does not match collection",
                ));
            }
            Ok(collection.clone())
        }
        Err(error) if error.code() == ErrorCode::CollectionNotFound => {
            let config = CollectionConfig {
                tokenizer: options.tokenizer.clone(),
                name: options.collection_name.clone(),
                embedding_dim: options.embedding_dim,
                distance_metric: options.distance_metric,
                embedding_model: options.embedding_model.clone(),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            };
            work.merge(config.validate_in(context)?);
            catalog.check_collection_name_available(&config.name)?;
            Ok(CollectionInfo {
                tokenizer: config.tokenizer,
                collection_id: catalog.next_id("col_", 0)?,
                name: config.name,
                embedding_dim: config.embedding_dim,
                distance_metric: config.distance_metric,
                embedding_model: config.embedding_model,
                vectors_normalized: config.vectors_normalized,
                vector_index: config.vector_index,
                created_at: time::now_rfc3339(),
            })
        }
        Err(error) => Err(error),
    }
}

fn replacement(kind: &str, id: &str, timestamp: &str) -> RecordPayload {
    RecordPayload::Tombstone(Tombstone {
        target_type: kind.into(),
        target_id: id.into(),
        deleted_at: timestamp.into(),
        reason: Some("replace".into()),
    })
}

fn existing_id(kind: &str, id: &str) -> Error {
    Error::new(
        ErrorCode::DuplicateExternalId,
        format!("{kind} {id} already exists"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::JsonInputKind;
    use crate::storage::scan_file;

    #[test]
    fn measured_chunk_batches_follow_byte_and_json_admission_limits() {
        for mode in [
            ImportIndexBuildMode::InlineFullCheckpoint,
            ImportIndexBuildMode::DeferredFullCheckpoint,
        ] {
            for json_pressure in [false, true] {
                let directory = std::env::temp_dir().join(format!(
                    "cofferdb-measured-batches-{mode:?}-{json_pressure}-{}",
                    std::process::id(),
                ));
                fs::create_dir(&directory).unwrap();
                let path = directory.join("data.coffer");
                let manifest = directory.join("manifest.json");
                fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
                fs::write(
                    directory.join("documents.jsonl"),
                    r#"{"external_id":"d","metadata":{}}"#,
                )
                .unwrap();
                let text = if json_pressure {
                    "needle".into()
                } else {
                    "needle ".repeat(8000)
                };
                let padding = if json_pressure {
                    "x".repeat(32 * 1024)
                } else {
                    String::new()
                };
                let chunks = (0..20).map(|id| format!("{{\"document_external_id\":\"d\",\"chunk_external_id\":\"c-{id}\",\"ordinal\":{id},\"text\":\"{text}\",\"metadata\":{{\"padding\":\"{padding}\"}}}}\n")).collect::<String>();
                fs::write(directory.join("chunks.jsonl"), chunks).unwrap();
                let db = Database::create(
                    &path,
                    DatabaseConfig {
                        cache_budget_mb: Some(if json_pressure { 64 } else { 1 }),
                        durability: Durability::Fast,
                        ..DatabaseConfig::default()
                    },
                )
                .unwrap();
                let context = db.json_context();
                let mut busy = context.reservation();
                if json_pressure {
                    busy.grow(
                        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 512 * 1024,
                        JsonInputKind::BenchmarkArtifact,
                    )
                    .unwrap();
                }
                let started = Instant::now();
                let measured = db
                    .import_manifest_measured_in(
                        &manifest,
                        OnConflict::Error,
                        mode,
                        context.clone(),
                    )
                    .unwrap();
                let elapsed = started.elapsed();
                assert_eq!(measured.summary.chunks_inserted, 20);
                let scan = scan_file(&path).unwrap();
                assert_eq!(measured.batches.committed_batches, scan.commits);
                assert_eq!(
                    measured.batches.checkpoint_cycles,
                    if mode == ImportIndexBuildMode::InlineFullCheckpoint {
                        scan.commits
                    } else {
                        1
                    }
                );
                assert_eq!(
                    measured.batches.checkpoint_ready_cycles,
                    measured.batches.checkpoint_cycles
                );
                assert!(measured.batches.final_checkpoint_ready);
                assert!(measured.batches.commit_time + measured.batches.checkpoint_time <= elapsed);
                let mut actual = BTreeMap::<u64, usize>::new();
                for record in scan.records {
                    if matches!(record.payload, RecordPayload::ChunkPut(_)) {
                        *actual.entry(record.record_ref.batch_id).or_default() += 1;
                    }
                }
                assert_eq!(actual.values().sum::<usize>(), 20);
                assert!(
                    actual.len() > 1,
                    "fixture must exercise actual admission splitting"
                );
                assert_eq!(
                    measured.batches.max_batch_chunks,
                    *actual.values().max().unwrap()
                );
                assert!(measured.batches.max_batch_chunks < 20);
                assert_eq!(db.temporary_disk_budget().used(), 0);
                drop(busy);
                drop(db);
                fs::remove_dir_all(directory).unwrap();
            }
        }
    }

    #[test]
    fn disk_reference_plans_preserve_mixed_skips_and_sparse_physical_lines() {
        let directory =
            std::env::temp_dir().join(format!("cofferdb-reference-mixed-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let manifest = directory.join("manifest.json");
        fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"embedding_model":"m","distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl"}"#).unwrap();
        let write_inputs = |initial: bool| {
            let mut documents = String::from("\n\n");
            let mut chunks = String::new();
            let mut embeddings = String::new();
            for order in 0..8 {
                let id = (order * 5 + 3) % 8;
                if !initial || id % 2 == 0 {
                    documents.push_str(&format!(
                        "{{\"external_id\":\"d-{id}\",\"metadata\":{{}}}}\n\n"
                    ));
                }
            }
            for order in 0..16 {
                let id = (order * 7 + 5) % 16;
                let document = id / 2;
                let ordinal = id % 2;
                if !initial || (document % 2 == 0 && ordinal == 0) {
                    chunks.push_str(&format!("{{\"document_external_id\":\"d-{document}\",\"chunk_external_id\":\"c-{id}\",\"ordinal\":{ordinal},\"text\":\"needle\",\"metadata\":{{}}}}\n\n"));
                    embeddings.insert_str(0, &format!("{{\"chunk_external_id\":\"c-{id}\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}}\n"));
                }
            }
            fs::write(directory.join("documents.jsonl"), documents).unwrap();
            fs::write(directory.join("chunks.jsonl"), chunks).unwrap();
            fs::write(directory.join("embeddings.jsonl"), embeddings).unwrap();
        };
        let path = directory.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        write_inputs(true);
        db.import_manifest(&manifest, OnConflict::Error).unwrap();
        write_inputs(false);
        let summary = db.import_manifest(&manifest, OnConflict::Skip).unwrap();
        assert_eq!(
            (summary.documents_inserted, summary.documents_skipped),
            (4, 4)
        );
        assert_eq!((summary.chunks_inserted, summary.chunks_skipped), (8, 8));
        assert_eq!(
            (summary.embeddings_inserted, summary.embeddings_skipped),
            (8, 8)
        );
        assert_eq!(db.temporary_disk_budget().used(), 0);
        drop(db);
        let db = Database::open_read_only(&path).unwrap();
        let collection = db.collection("notes").unwrap();
        for id in 0..16 {
            let chunk = collection.get_chunk_by_external_id(&format!("c-{id}"));
            if id / 2 % 2 == 0 && id % 2 == 1 {
                assert_eq!(chunk.unwrap_err().code(), ErrorCode::ChunkNotFound);
                continue;
            }
            let chunk = chunk.unwrap();
            let document = collection.get_document(&chunk.document_id).unwrap();
            assert_eq!(document.external_id, format!("d-{}", id / 2));
            assert_eq!(chunk.ordinal, id % 2);
            assert!(chunk.embedding_id.is_some());
        }
        drop(collection);
        drop(db);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn deferred_reference_duplicates_report_the_source_file_before_any_commit() {
        for duplicate_documents in [false, true] {
            for include_chunks in [false, true] {
                if !duplicate_documents && !include_chunks {
                    continue;
                }
                let directory = std::env::temp_dir().join(format!(
                    "cofferdb-reference-duplicates-{duplicate_documents}-{include_chunks}-{}",
                    std::process::id()
                ));
                fs::create_dir(&directory).unwrap();
                let manifest = directory.join("manifest.json");
                fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
                let prefix = "x".repeat(256);
                let documents = (0..200)
                    .map(|id| format!("{{\"external_id\":\"{prefix}d-{id}\",\"metadata\":{{}}}}\n"))
                    .collect::<String>();
                let chunks = if include_chunks {
                    (0..200).map(|id| format!("{{\"document_external_id\":\"{prefix}d-{id}\",\"chunk_external_id\":\"{prefix}c-{id}\",\"ordinal\":0,\"text\":\"needle\",\"metadata\":{{}}}}\n")).collect::<String>()
                } else {
                    String::new()
                };
                let document_path = directory.join("documents.jsonl");
                let chunk_path = directory.join("chunks.jsonl");
                fs::write(&document_path, &documents).unwrap();
                fs::write(&chunk_path, &chunks).unwrap();
                let path = directory.join("data.coffer");
                let db = Database::create(
                    &path,
                    DatabaseConfig {
                        cache_budget_mb: Some(8),
                        ..DatabaseConfig::default()
                    },
                )
                .unwrap();
                for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
                    let before = fs::read(&path).unwrap();
                    let (source, text) = if duplicate_documents {
                        (&document_path, &documents)
                    } else {
                        (&chunk_path, &chunks)
                    };
                    let duplicate = text.lines().next().unwrap();
                    let duplicate = if duplicate_documents {
                        duplicate.to_owned()
                    } else {
                        duplicate.replace("\"ordinal\":0", "\"ordinal\":1")
                    };
                    fs::write(source, format!("{text}\n{duplicate}\n")).unwrap();
                    let error = db.import_manifest(&manifest, conflict).unwrap_err();
                    assert_eq!(error.code(), ErrorCode::DuplicateExternalId, "{error:?}");
                    assert!(error.detail().iter().any(
                        |(key, value)| key == "path" && Some(value.as_str()) == source.to_str()
                    ));
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "line" && value == "202"));
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "first_line" && value == "1"));
                    assert_eq!(fs::read(&path).unwrap(), before);
                    assert_eq!(db.temporary_disk_budget().used(), 0);
                    fs::write(source, text).unwrap();
                    if conflict == OnConflict::Error {
                        db.import_manifest(&manifest, conflict).unwrap();
                    }
                }
                drop(db);
                fs::remove_dir_all(directory).unwrap();
            }
        }
    }

    #[test]
    fn manifest_stream_keeps_interleaved_document_relationships_across_one_hundred_orders() {
        let directory = std::env::temp_dir().join(format!(
            "cofferdb-streamed-manifest-orders-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let manifest = directory.join("manifest.json");
        fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
        for seed in 0..100 {
            let documents = (0..4).map(|index| {
                let id = (index + seed) % 4;
                format!("{{\"external_id\":\"d-{id}\",\"title\":\"title {id}\",\"metadata\":{{\"id\":{id}}}}}\n")
            }).collect::<String>();
            let chunks = (0..12).map(|index| {
                let id = (index * 5 + seed) % 12;
                let document = id / 3;
                let ordinal = id % 3;
                format!("{{\"document_external_id\":\"d-{document}\",\"chunk_external_id\":\"c-{id}\",\"ordinal\":{ordinal},\"text\":\"needle {id}\",\"metadata\":{{\"id\":{id}}},\"source_span\":{{\"start\":0,\"end\":6}}}}\n")
            }).collect::<String>();
            fs::write(directory.join("documents.jsonl"), documents).unwrap();
            fs::write(directory.join("chunks.jsonl"), chunks).unwrap();
            let path = directory.join(format!("data-{seed}.coffer"));
            let db = Database::create(
                &path,
                DatabaseConfig {
                    durability: Durability::Fast,
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
                let before = fs::read(&path).unwrap();
                let measured = db
                    .import_manifest_measured_in(
                        &manifest,
                        conflict,
                        ImportIndexBuildMode::DeferredFullCheckpoint,
                        db.json_context(),
                    )
                    .unwrap();
                assert_eq!(
                    measured.batches.max_batch_chunks,
                    if conflict == OnConflict::Skip { 0 } else { 12 },
                );
                let summary = measured.summary;
                assert_eq!((summary.documents_seen, summary.chunks_seen), (4, 12));
                if conflict == OnConflict::Skip {
                    assert_eq!((summary.documents_skipped, summary.chunks_skipped), (4, 12));
                    assert_eq!(fs::read(&path).unwrap(), before);
                } else {
                    assert_eq!(
                        (summary.documents_inserted, summary.chunks_inserted),
                        (4, 12)
                    );
                }
            }
            drop(db);
            let db = Database::open_read_only(&path).unwrap();
            let collection = db.collection("notes").unwrap();
            for id in 0..12 {
                let chunk = collection
                    .get_chunk_by_external_id(&format!("c-{id}"))
                    .unwrap();
                let document = collection.get_document(&chunk.document_id).unwrap();
                assert_eq!(document.external_id, format!("d-{}", id / 3));
                assert_eq!(
                    document.title.as_deref(),
                    Some(format!("title {}", id / 3).as_str())
                );
                assert_eq!(
                    document.metadata.get("id").and_then(JsonValue::as_f64),
                    Some((id / 3) as f64)
                );
                assert_eq!(chunk.ordinal, id % 3);
                assert_eq!(chunk.text, format!("needle {id}"));
                assert_eq!(
                    chunk.metadata.get("id").and_then(JsonValue::as_f64),
                    Some(id as f64)
                );
                assert_eq!(chunk.source_span.as_ref().unwrap().end, Some(6));
            }
            let hits = collection
                .search()
                .text("needle")
                .top_k(20)
                .execute()
                .unwrap();
            assert_eq!(hits.results.len(), 12);
            drop(collection);
            drop(db);
        }
        fs::remove_dir_all(directory).unwrap();
    }
}
