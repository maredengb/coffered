use super::*;
use crate::json::{JsonOperationContext, ReservedString};
use crate::model::ChunkInput;
use crate::storage::{PreparedRecordsBuilder, RecordReservation};
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
    pub vectors_normalized: Option<bool>,
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
        source_span_metadata_conflict: bool,
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
        batch_target: usize,
        context: &JsonOperationContext,
        message: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            sorting: Some(ExternalUniqueKeys::new_growing(
                &std::env::temp_dir(),
                database.temporary_disk_budget(),
                batch_target,
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
        warnings: &mut crate::import::warnings::ImportWarnings,
        visit: impl FnOnce(&mut dyn FnMut(ImportRow, RecordReservation) -> Result<()>) -> Result<()>,
    ) -> Result<(InsertSummary, ImportBatchMetrics)> {
        let mut inner = self.lock()?;
        let millis = time::unix_millis_now();
        let now = inner.read_reclaiming(|_| {
            ReservedString::build(&context, "batch_prepare", |out| {
                time::write_unix_millis(out, millis)
            })
        })?;
        let mut collection_work = RecordReservation::from(context.reservation());
        let (collection, work) = inner.read_reclaiming(|inner| {
            import_collection(&inner.catalog, &options, &context, now.as_ref())
        })?;
        collection_work.merge(work);
        let mut models = inner
            .read_reclaiming(|inner| ModelAdmission::new(&inner.catalog, &collection, &context))?;
        let mut preflight_error =
            validate_collection_target(&inner.catalog, &collection, &options).err();
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
        if preflight_error.is_none()
            && !inner
                .catalog
                .collections
                .contains_key(&collection.collection_id)
        {
            let (created, mut work) =
                inner.read_reclaiming(|_| copy_collection_in(&collection, &context))?;
            records.push_retained(RecordPayload::CollectionCreate(created), &mut work.json)?;
        }

        let mut summary = InsertSummary::default();
        let mut ordinal_keys: Option<ExternalUniqueKeys> = None;
        let mut ordinal_error = None;
        let mut embedding_keys: Option<ExternalUniqueKeys> = None;
        let reference_batch_target = (inner.active_cache_budget_mb().saturating_mul(1_048_576)
            / 128)
            .clamp(64 * 1024, 1_048_576);
        let mut documents = ImportReferences::new(
            self,
            &options.document_path,
            reference_batch_target,
            &context,
            "document external_id appears more than once in the same import batch",
        )?;
        let mut chunks = ImportReferences::new(
            self,
            &options.chunk_path,
            reference_batch_target,
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
                    if preflight_error.is_some() {
                        documents.insert(&input.external_id, line, 0)?;
                        return Ok(());
                    }
                    let existing = inner
                        .catalog
                        .document_external_ids
                        .get(&collection.collection_id, &input.external_id)
                        .filter(|id| inner.catalog.is_document_live(id));
                    if existing.is_some() {
                        match options.on_conflict {
                            OnConflict::Error => {
                                documents.insert(&input.external_id, line, 0)?;
                                preflight_error = Some(crate::import::with_jsonl_location(
                                    existing_id("document external_id", &input.external_id),
                                    &options.document_path,
                                    line,
                                ));
                                return Ok(());
                            }
                            OnConflict::Skip => {
                                summary.documents_skipped += 1;
                                documents.insert(&input.external_id, line, 0)?;
                                return Ok(());
                            }
                            OnConflict::Replace => {
                                let (record, mut work) = inner.read_reclaiming(|inner| {
                                    let document = inner.catalog.document_by_external_id(
                                        &collection.collection_id,
                                        &input.external_id,
                                    )?;
                                    replacement(
                                        "document",
                                        &document.document_id,
                                        now.as_ref(),
                                        &context,
                                    )
                                })?;
                                records.push_retained(record, &mut work.json)?;
                                summary.documents_replaced += 1;
                            }
                        }
                    }
                    let (fields_work, [document_id, collection_id, created_at, updated_at]) = inner
                        .read_reclaiming(|inner| {
                            let mut work = RecordReservation::from(context.reservation());
                            let (document_id, id_work) = inner
                                .catalog
                                .next_id_in("doc_", summary.documents_inserted as u64, &context)?
                                .into_parts();
                            work.merge_workspace(id_work);
                            let [collection_id, created_at, updated_at] = work.copy_strings(
                                [&collection.collection_id, now.as_ref(), now.as_ref()],
                                &context,
                            )?;
                            Ok((work, [document_id, collection_id, created_at, updated_at]))
                        })?;
                    documents.insert(
                        &input.external_id,
                        line,
                        summary.documents_inserted as u64 + 1,
                    )?;
                    records.push_retained(
                        RecordPayload::DocumentPut(Document {
                            document_id,
                            collection_id,
                            external_id: input.external_id,
                            title: input.title,
                            source_uri: input.source_uri,
                            metadata: input.metadata,
                            created_at,
                            updated_at,
                        }),
                        &mut row_work.json,
                    )?;
                    drop(fields_work);
                    summary.documents_inserted += 1;
                }
                ImportRow::Chunk {
                    document_external_id,
                    chunk,
                    line,
                    source_span_metadata_conflict,
                } => {
                    if source_span_metadata_conflict {
                        inner.read_reclaiming(|_| {
                            warnings
                                .push(crate::import::warnings::ImportWarnings::SOURCE_SPAN_CONFLICT)
                        })?;
                    }
                    chunk.validate_preprocessed()?;
                    if ordinal_keys.is_none() && ordinal_error.is_none() {
                        ordinal_keys = Some(ExternalUniqueKeys::new_growing(
                            &std::env::temp_dir(),
                            self.temporary_disk_budget(),
                            reference_batch_target,
                            &context,
                            ErrorCode::InvalidJsonlSchema,
                            "chunk ordinal must be unique within a document",
                        )?);
                    }
                    if let Some(keys) = ordinal_keys.as_mut() {
                        match keys.push(
                            &[
                                document_external_id.as_bytes(),
                                &(chunk.ordinal as u64).to_le_bytes(),
                            ],
                            line as u64,
                        ) {
                            Err(error) if error.code() == ErrorCode::InvalidJsonlSchema => {
                                ordinal_error =
                                    Some(error.with_detail("path", options.chunk_path.display()));
                                drop(ordinal_keys.take());
                            }
                            result => result?,
                        }
                    }
                    if preflight_error.is_some() || ordinal_error.is_some() {
                        chunks.insert(&chunk.chunk_external_id, line, 0)?;
                        return Ok(());
                    }
                    let (_, planned_document) = match documents.get(&document_external_id)? {
                        Some(reference) => reference,
                        None => {
                            chunks.insert(&chunk.chunk_external_id, line, 0)?;
                            preflight_error = Some(crate::import::with_jsonl_location(
                                Error::new(ErrorCode::InvalidJsonlSchema,
                                    format!("chunks reference missing document_external_id {document_external_id}")),
                                &options.chunk_path,
                                line,
                            ));
                            return Ok(());
                        }
                    };
                    if planned_document == 0 {
                        chunks.insert(&chunk.chunk_external_id, line, 0)?;
                        summary.chunks_skipped += 1;
                        return Ok(());
                    }
                    let existing = inner
                        .catalog
                        .chunk_external_ids
                        .get(&collection.collection_id, &chunk.chunk_external_id)
                        .filter(|id| inner.catalog.is_chunk_live(id));
                    if existing.is_some() {
                        match options.on_conflict {
                            OnConflict::Error => {
                                chunks.insert(&chunk.chunk_external_id, line, 0)?;
                                preflight_error = Some(crate::import::with_jsonl_location(
                                    existing_id("chunk_external_id", &chunk.chunk_external_id),
                                    &options.chunk_path,
                                    line,
                                ));
                                return Ok(());
                            }
                            OnConflict::Skip => {
                                chunks.insert(&chunk.chunk_external_id, line, 0)?;
                                summary.chunks_skipped += 1;
                                return Ok(());
                            }
                            OnConflict::Replace => {
                                let (record, mut work) = inner.read_reclaiming(|inner| {
                                    let chunk = inner.catalog.chunk_by_external_id(
                                        &collection.collection_id,
                                        &chunk.chunk_external_id,
                                    )?;
                                    replacement("chunk", &chunk.chunk_id, now.as_ref(), &context)
                                })?;
                                records.push_retained(record, &mut work.json)?;
                                summary.chunks_replaced += 1;
                            }
                        }
                    }
                    let source_span = chunk.resolved_source_span()?;
                    let token_count =
                        inner.read_reclaiming(|_| token_count_hint_in(&chunk.text, &context))?;
                    let (
                        fields_work,
                        [chunk_id, document_id, collection_id, text_hash, created_at],
                    ) = inner.read_reclaiming(|inner| {
                        let mut work = RecordReservation::from(context.reservation());
                        let (chunk_id, id_work) = inner
                            .catalog
                            .next_id_in("chk_", summary.chunks_inserted as u64, &context)?
                            .into_parts();
                        work.merge_workspace(id_work);
                        let (document_id, id_work) = inner
                            .catalog
                            .next_id_in("doc_", planned_document - 1, &context)?
                            .into_parts();
                        work.merge_workspace(id_work);
                        let (text_hash, hash_work) =
                            ReservedString::build_exact(64, &context, "batch_prepare", |out| {
                                crate::hash::write_digest_hex(
                                    out,
                                    crate::hash::sha256(chunk.text.as_bytes()),
                                )
                            })?
                            .into_parts();
                        work.merge_workspace(hash_work);
                        let [collection_id, created_at] =
                            work.copy_strings([&collection.collection_id, now.as_ref()], &context)?;
                        Ok((
                            work,
                            [chunk_id, document_id, collection_id, text_hash, created_at],
                        ))
                    })?;
                    chunks.insert(
                        &chunk.chunk_external_id,
                        line,
                        summary.chunks_inserted as u64 + 1,
                    )?;
                    records.push_retained(
                        RecordPayload::ChunkPut(Chunk {
                            chunk_id,
                            document_id,
                            collection_id,
                            chunk_external_id: chunk.chunk_external_id,
                            text_hash,
                            token_count,
                            text: chunk.text,
                            ordinal: chunk.ordinal,
                            metadata: chunk.metadata,
                            embedding_id: None,
                            source_span,
                            created_at,
                        }),
                        &mut row_work.json,
                    )?;
                    drop(fields_work);
                    summary.chunks_inserted += 1;
                }
                ImportRow::Embedding { input, line } => {
                    documents.finish()?;
                    chunks.finish()?;
                    finish_ordinal_keys(
                        &mut ordinal_keys,
                        &options.chunk_path,
                        &mut ordinal_error,
                    )?;
                    if embedding_keys.is_none() {
                        embedding_keys = Some(ExternalUniqueKeys::new_growing(
                            &std::env::temp_dir(), self.temporary_disk_budget(), reference_batch_target,
                            &context,
                            ErrorCode::DuplicateExternalId,
                            "embedding chunk_external_id appears more than once in the same import batch",
                        )?);
                    }
                    embedding_keys
                        .as_mut()
                        .unwrap()
                        .push(&[input.chunk_external_id.as_bytes()], line as u64)?;
                    input.validate_structure()?;
                    if preflight_error.is_some() || ordinal_error.is_some() {
                        return Ok(());
                    }
                    if let Err(error) = input.validate_target_dimension(options.embedding_dim) {
                        preflight_error = Some(match options.embedding_path.as_deref() {
                            Some(path) => crate::import::with_jsonl_location(error, path, line),
                            None => error,
                        });
                        return Ok(());
                    }
                    let (_, planned) = match chunks.get(&input.chunk_external_id)? {
                        Some(reference) => reference,
                        None => {
                            let error = Error::new(ErrorCode::InvalidJsonlSchema,
                                format!("embeddings reference chunk_external_id {} not present in this manifest; use import-embeddings for embedding-only updates", input.chunk_external_id));
                            preflight_error = Some(match options.embedding_path.as_deref() {
                                Some(path) => crate::import::with_jsonl_location(error, path, line),
                                None => error,
                            });
                            return Ok(());
                        }
                    };
                    let fields = inner.read_reclaiming(|inner| {
                        let mut work = RecordReservation::from(context.reservation());
                        let chunk_id = match planned {
                            0 => match inner.catalog.chunk_by_external_id(
                                &collection.collection_id,
                                &input.chunk_external_id,
                            ) {
                                Ok(chunk) => {
                                    if options.on_conflict == OnConflict::Skip
                                        && inner
                                            .catalog
                                            .embedding_for_chunk(&chunk.chunk_id)
                                            .is_some()
                                    {
                                        return Ok(None);
                                    }
                                    work.copy_string(&chunk.chunk_id, &context)?
                                }
                                Err(error) => return Err(error),
                            },
                            number => {
                                let (id, id_work) = inner
                                    .catalog
                                    .next_id_in("chk_", number - 1, &context)?
                                    .into_parts();
                                work.merge_workspace(id_work);
                                id
                            }
                        };
                        let replaced = inner.catalog.embedding_for_chunk(&chunk_id).is_some();
                        if replaced {
                            match options.on_conflict {
                                OnConflict::Error => {
                                    return Err(existing_id(
                                        "embedding for chunk",
                                        &input.chunk_external_id,
                                    ))
                                }
                                OnConflict::Skip => return Ok(None),
                                OnConflict::Replace => {}
                            }
                        }
                        let (embedding_id, id_work) = inner
                            .catalog
                            .next_id_in("emb_", summary.embeddings_inserted as u64, &context)?
                            .into_parts();
                        work.merge_workspace(id_work);
                        let [collection_id, created_at] =
                            work.copy_strings([&collection.collection_id, now.as_ref()], &context)?;
                        Ok(Some((
                            work,
                            [embedding_id, chunk_id, collection_id, created_at],
                            replaced,
                        )))
                    });
                    let fields = match fields {
                        Err(error)
                            if matches!(
                                error.code(),
                                ErrorCode::ChunkNotFound | ErrorCode::DuplicateExternalId
                            ) =>
                        {
                            preflight_error = Some(match options.embedding_path.as_deref() {
                                Some(path) => crate::import::with_jsonl_location(error, path, line),
                                None => error,
                            });
                            return Ok(());
                        }
                        result => result?,
                    };
                    let Some((
                        fields_work,
                        [embedding_id, chunk_id, collection_id, created_at],
                        replaced,
                    )) = fields
                    else {
                        summary.embeddings_skipped += 1;
                        return Ok(());
                    };
                    match inner.read_reclaiming(|_| {
                        models.validate_write(
                            &input.model_name,
                            options.embedding_model.as_deref(),
                            &context,
                        )
                    }) {
                        Err(error) if error.code() == ErrorCode::EmbeddingModelMismatch => {
                            preflight_error = Some(match options.embedding_path.as_deref() {
                                Some(path) => crate::import::with_jsonl_location(error, path, line),
                                None => error,
                            });
                            return Ok(());
                        }
                        result => result?,
                    }
                    records.push_retained(
                        RecordPayload::EmbeddingPut(Embedding {
                            embedding_id,
                            chunk_id,
                            collection_id,
                            chunk_external_id: input.chunk_external_id,
                            model_name: input.model_name,
                            dim: input.dim,
                            vector: input.vector,
                            created_at,
                        }),
                        &mut row_work.json,
                    )?;
                    drop(fields_work);
                    summary.embeddings_replaced += usize::from(replaced);
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
        finish_reference_keys(&mut embedding_keys, options.embedding_path.as_deref())?;
        finish_ordinal_keys(&mut ordinal_keys, &options.chunk_path, &mut ordinal_error)?;
        if let Some(error) = ordinal_error {
            return Err(error);
        }
        if let Some(error) = preflight_error {
            return Err(error);
        }
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

fn finish_ordinal_keys(
    keys: &mut Option<ExternalUniqueKeys>,
    path: &Path,
    pending: &mut Option<Error>,
) -> Result<()> {
    let result = finish_reference_keys(keys, Some(path));
    match result {
        Err(error) if error.code() == ErrorCode::InvalidJsonlSchema => {
            *pending = Some(error);
            Ok(())
        }
        result => result,
    }
}

fn validate_collection_target(
    catalog: &Catalog,
    collection: &CollectionInfo,
    options: &ManifestImportOptions,
) -> Result<()> {
    if !catalog.collections.contains_key(&collection.collection_id) {
        return catalog.check_collection_name_available(&collection.name);
    }
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
    if options
        .vectors_normalized
        .is_some_and(|value| value != collection.vectors_normalized)
    {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "manifest vectors_normalized does not match collection configuration",
        )
        .with_detail("field", "vectors_normalized"));
    }
    if let (Some(expected), Some(actual)) = (&collection.embedding_model, &options.embedding_model)
    {
        if expected != actual {
            return Err(Error::new(
                ErrorCode::EmbeddingModelMismatch,
                "manifest embedding_model does not match the explicit collection configuration",
            )
            .with_detail("expected", expected)
            .with_detail("actual", actual));
        }
    }
    Ok(())
}

fn import_collection(
    catalog: &Catalog,
    options: &ManifestImportOptions,
    context: &JsonOperationContext,
    timestamp: &str,
) -> Result<(CollectionInfo, RecordReservation)> {
    match catalog.collection_by_name(&options.collection_name) {
        Ok(collection) => copy_collection_in(collection, context),
        Err(error) if error.code() == ErrorCode::CollectionNotFound => {
            let mut work = RecordReservation::from(context.reservation());
            let [name, embedding_model, created_at] = work.copy_strings(
                [
                    &options.collection_name,
                    options.embedding_model.as_deref().unwrap_or_default(),
                    timestamp,
                ],
                context,
            )?;
            let config = CollectionConfig {
                tokenizer: options.tokenizer.clone(),
                name,
                embedding_dim: options.embedding_dim,
                distance_metric: options.distance_metric,
                embedding_model: options.embedding_model.as_ref().map(|_| embedding_model),
                vectors_normalized: options.vectors_normalized.unwrap_or(false),
                vector_index: VectorIndexConfig::Exact,
            };
            let string_bytes = config.name.capacity()
                + config.embedding_model.as_ref().map_or(0, String::capacity);
            // Reuse these strings during schema validation, then restore their typed lifetime.
            let (json, workspace) = work.parts_mut();
            json.merge(
                workspace
                    .unwrap()
                    .take_json(string_bytes, crate::json::JsonInputKind::CollectionConfig)?,
            );
            let validation = config.validate_retained_in(&work.json);
            let strings = work.json.take_workspace(string_bytes);
            work.merge_workspace(strings);
            drop(validation?);
            let (collection_id, id_work) = catalog.next_id_in("col_", 0, context)?.into_parts();
            work.merge_workspace(id_work);
            Ok((
                CollectionInfo {
                    tokenizer: config.tokenizer,
                    collection_id,
                    name: config.name,
                    embedding_dim: config.embedding_dim,
                    distance_metric: config.distance_metric,
                    embedding_model: config.embedding_model,
                    vectors_normalized: config.vectors_normalized,
                    vector_index: config.vector_index,
                    created_at,
                },
                work,
            ))
        }
        Err(error) => Err(error),
    }
}

fn replacement(
    kind: &str,
    id: &str,
    timestamp: &str,
    context: &JsonOperationContext,
) -> Result<(RecordPayload, RecordReservation)> {
    let mut work = RecordReservation::from(context.reservation());
    let [target_type, target_id, deleted_at, reason] =
        work.copy_strings([kind, id, timestamp, "replace"], context)?;
    Ok((
        RecordPayload::Tombstone(Tombstone {
            target_type,
            target_id,
            deleted_at,
            reason: Some(reason),
        }),
        work,
    ))
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

    fn field_options() -> ManifestImportOptions {
        ManifestImportOptions {
            tokenizer: Default::default(),
            collection_name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".into()),
            vectors_normalized: None,
            document_path: "documents.jsonl".into(),
            chunk_path: "chunks.jsonl".into(),
            embedding_path: Some("embeddings.jsonl".into()),
            on_conflict: OnConflict::Error,
            index_build_mode: ImportIndexBuildMode::DeferredFullCheckpoint,
        }
    }

    #[test]
    fn manifest_collection_fields_keep_their_aggregate_credit_and_refund_failed_copies() {
        let mut options = field_options();
        options.collection_name = "n".repeat(65);
        options.embedding_model = Some("m".repeat(113));
        let timestamp = "2026-10-11T04:00:00.000Z";
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut catalog = Catalog::default();
        let (collection, work) =
            import_collection(&catalog, &options, &context, timestamp).unwrap();
        let needed = work.json_bytes() + work.workspace_bytes();
        let ordinary = collection.collection_id.capacity()
            + collection.name.capacity()
            + collection.embedding_model.as_ref().unwrap().capacity()
            + collection.created_at.capacity();
        assert_eq!(work.workspace_bytes(), ordinary);
        assert_eq!(work.json_bytes(), 0);
        assert_eq!(context.available_workspace_bytes(), available - needed);
        catalog
            .collection_names
            .insert(collection.name.clone(), collection.collection_id.clone());
        catalog
            .collections
            .insert(collection.collection_id.clone(), collection.clone());
        drop((collection, work));
        assert_eq!(context.available_workspace_bytes(), available);
        for existing in [false, true] {
            let source = if existing {
                &catalog
            } else {
                &Catalog::default()
            };
            let error = crate::test_alloc::fail_byte_allocation_once(113, || {
                import_collection(source, &options, &context, timestamp)
            })
            .err()
            .unwrap();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert_eq!(context.available_workspace_bytes(), available);
            let (collection, work) =
                import_collection(source, &options, &context, timestamp).unwrap();
            assert_eq!(work.workspace_bytes(), ordinary);
            if existing {
                assert_eq!(work.json_bytes(), 0);
            }
            drop((collection, work));
            assert_eq!(context.available_workspace_bytes(), available);
        }
        let first_copy = options.collection_name.len()
            + options.embedding_model.as_ref().unwrap().len()
            + timestamp.len();
        for parent in [false, true] {
            let limit = first_copy - 1;
            let context = JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(
                    if parent { 2 * first_copy } else { limit },
                    0,
                    if parent { limit } else { 2 * first_copy },
                ),
            ));
            let result = import_collection(&Catalog::default(), &options, &context, timestamp);
            let error = result.err().unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            for (key, expected) in [
                ("resource", "memory".to_owned()),
                ("operation", "operation_scratch".to_owned()),
                ("limit_bytes", limit.to_string()),
            ] {
                assert!(error
                    .detail()
                    .iter()
                    .any(|(field, value)| field == key && value == &expected));
            }
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "requested_bytes" && value == &first_copy.to_string()));
            assert_eq!(context.available_workspace_bytes(), limit);
            assert_eq!(context.available_catalog_bytes(), 0);
        }
    }

    #[test]
    fn manifest_replacement_fields_are_admitted_once_and_native_failures_are_not_budget_errors() {
        let id = "d".repeat(257);
        let timestamp = "2026-10-11T04:00:00.000Z";
        let needed = "document".len() + id.len() + timestamp.len() + "replace".len();
        for exact in [false, true] {
            let limit = needed - usize::from(!exact);
            let context = JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(limit, 0, 2 * needed),
            ));
            let result = replacement("document", &id, timestamp, &context);
            if exact {
                let (record, work) = result.unwrap();
                let RecordPayload::Tombstone(tombstone) = &record else {
                    panic!("replacement kind")
                };
                assert_eq!(tombstone.target_type, "document");
                assert_eq!(tombstone.target_id, id);
                assert_eq!(tombstone.deleted_at, timestamp);
                assert_eq!(tombstone.reason.as_deref(), Some("replace"));
                assert_eq!(work.workspace_bytes(), needed);
                assert_eq!(context.available_workspace_bytes(), 0);
                drop((record, work));
            } else {
                assert_eq!(result.err().unwrap().code(), ErrorCode::ResourceExhausted);
            }
            assert_eq!(context.available_workspace_bytes(), limit);
        }
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let error = crate::test_alloc::fail_byte_allocation_once(id.len(), || {
            replacement("document", &id, timestamp, &context)
        })
        .err()
        .unwrap();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(context.available_workspace_bytes(), available);
    }

    #[test]
    fn manifest_collection_validation_restores_string_credit_before_returning_errors() {
        let context = JsonOperationContext::standalone();
        let initial = (
            context.available_workspace_bytes(),
            context.available_total_bytes(),
        );
        for problem in ["name", "model", "dimension", "unit_bytes"] {
            let mut options = field_options();
            let expected = match problem {
                "name" => {
                    options.collection_name.clear();
                    ErrorCode::MissingRequiredField
                }
                "model" => {
                    options.embedding_model = Some(String::new());
                    ErrorCode::MissingRequiredField
                }
                "dimension" => {
                    options.embedding_dim = 0;
                    ErrorCode::InvalidEmbeddingDimension
                }
                _ => {
                    options.collection_name =
                        "n".repeat(crate::json::JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES);
                    ErrorCode::InvalidJsonlSchema
                }
            };
            let error = import_collection(
                &Catalog::default(),
                &options,
                &context,
                "2026-10-11T04:00:00.000Z",
            )
            .err()
            .unwrap();
            assert_eq!(error.code(), expected, "{problem}: {error}");
            if problem == "unit_bytes" {
                assert!(error.detail().iter().any(
                    |(key, value)| key == "json_limit_reason" && value == "unit_bytes_exceeded"
                ));
            }
            assert_eq!(
                (
                    context.available_workspace_bytes(),
                    context.available_total_bytes()
                ),
                initial
            );
        }
    }

    #[test]
    fn manifest_existing_embedding_skip_does_not_copy_its_internal_id() {
        let path = std::env::temp_dir().join(format!(
            "cofferdb-manifest-id-skip-{}.coffer",
            std::process::id()
        ));
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                embedding_model: Some("m".into()),
                distance_metric: DistanceMetric::Cosine,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "chunk".into(),
                    text: "needle".into(),
                    ordinal: 0,
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            })
            .unwrap();
        collection
            .put_embeddings(vec![EmbeddingInput {
                chunk_external_id: "chunk".into(),
                model_name: "m".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            }])
            .unwrap();
        let id_bytes = collection
            .get_chunk_by_external_id("chunk")
            .unwrap()
            .chunk_id
            .len();
        let before = fs::read(&path).unwrap();
        let context = db.json_context();
        let initial = context.available_workspace_bytes();
        let mut warnings = crate::import::warnings::ImportWarnings::new(&context).unwrap();
        let mut options = field_options();
        options.on_conflict = OnConflict::Skip;
        let (summary, metrics) = db
            .import_preprocessed_rows(options, context.clone(), &mut warnings, |emit| {
                emit(
                    ImportRow::Document {
                        input: ImportDocument {
                            external_id: "doc".into(),
                            title: None,
                            source_uri: None,
                            metadata: JsonValue::object(),
                        },
                        line: 1,
                    },
                    context.reservation().into(),
                )?;
                emit(
                    ImportRow::Chunk {
                        document_external_id: "doc".into(),
                        chunk: ChunkInput {
                            chunk_external_id: "chunk".into(),
                            text: "needle".into(),
                            ordinal: 0,
                            metadata: JsonValue::object(),
                            source_span: None,
                        },
                        line: 1,
                        source_span_metadata_conflict: false,
                    },
                    context.reservation().into(),
                )?;
                let row = ImportRow::Embedding {
                    input: EmbeddingInput {
                        chunk_external_id: "chunk".into(),
                        model_name: "m".into(),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    },
                    line: 1,
                };
                let (result, keys) = crate::test_alloc::key_allocations(id_bytes, || {
                    emit(row, context.reservation().into())
                });
                assert_eq!(keys.allocated, 0);
                assert!(!keys.overflow);
                result
            })
            .unwrap();
        assert_eq!(
            (
                summary.documents_skipped,
                summary.chunks_skipped,
                summary.embeddings_skipped
            ),
            (1, 1, 1)
        );
        assert_eq!(
            (
                summary.documents_inserted,
                summary.chunks_inserted,
                summary.embeddings_inserted
            ),
            (0, 0, 0)
        );
        assert_eq!(metrics.committed_batches, 0);
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(db.temporary_disk_budget().used(), 0);
        let (values, work) = warnings.into_parts();
        drop((values, work));
        assert_eq!(context.available_workspace_bytes(), initial);
        drop((collection, db));
        fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn manifest_generated_field_failure_targets_remain_valid_with_a_64_byte_tmpdir() {
        let parent = std::env::temp_dir();
        let mut name = format!("cofferdb-manifest-field-worker-{}-", std::process::id());
        name.push_str(&"x".repeat(64 - name.len()));
        assert_eq!(name.len(), 64);
        let directory = parent.join(&name);
        fs::create_dir(&directory).unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "db::import::tests::manifest_generated_field_failure_discards_the_staged_prefix_without_committing",
                "--nocapture",
            ])
            .current_dir(&parent)
            .env("TMPDIR", &name)
            .env("COFFERDB_MANIFEST_FAILURE_64_BYTE_TMPDIR", "1")
            .output();
        fs::remove_dir_all(directory).unwrap();
        let output = result.unwrap();
        assert!(
            output.status.success(),
            "generated-field worker failed: {:?}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[test]
    fn manifest_generated_field_failure_discards_the_staged_prefix_without_committing() {
        if std::env::var_os("COFFERDB_MANIFEST_FAILURE_64_BYTE_TMPDIR").is_some() {
            assert_eq!(
                std::env::temp_dir().as_os_str().as_encoded_bytes().len(),
                64
            );
        }
        let id_bytes = Catalog::default().next_id("doc_", 0).unwrap().len();
        for phase in ["document", "chunk", "embedding"] {
            let path = std::env::temp_dir().join(format!(
                "cofferdb-manifest-fields-{phase}-{}.coffer",
                std::process::id()
            ));
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let before = fs::read(&path).unwrap();
            let context = db.json_context();
            let available = (
                context.available_workspace_bytes(),
                context.available_catalog_bytes(),
                context.available_total_bytes(),
            );
            let mut warnings = crate::import::warnings::ImportWarnings::new(&context).unwrap();
            let mut visits = 0;
            let result = db.import_preprocessed_rows(
                field_options(),
                context.clone(),
                &mut warnings,
                |emit| {
                    let mut work = RecordReservation::from(context.reservation());
                    let external_id = work.copy_string("doc", &context)?;
                    let row = ImportRow::Document {
                        input: ImportDocument {
                            external_id,
                            title: None,
                            source_uri: None,
                            metadata: JsonValue::object(),
                        },
                        line: 1,
                    };
                    visits += 1;
                    if phase == "document" {
                        return crate::test_alloc::fail_byte_allocation_once(id_bytes, || {
                            emit(row, work)
                        });
                    }
                    emit(row, work)?;
                    if phase == "chunk" {
                        // Create ordinal/reference temporaries before the failure window:
                        // the 64-byte refusal must target the generated chunk checksum.
                        let mut prefix_work = RecordReservation::from(context.reservation());
                        let [document_external_id, chunk_external_id, text] = prefix_work
                            .copy_strings(["doc", "prefix-chunk", "prefix"], &context)?;
                        visits += 1;
                        emit(
                            ImportRow::Chunk {
                                document_external_id,
                                chunk: ChunkInput {
                                    chunk_external_id,
                                    text,
                                    ordinal: 0,
                                    metadata: JsonValue::object(),
                                    source_span: None,
                                },
                                line: 1,
                                source_span_metadata_conflict: false,
                            },
                            prefix_work,
                        )?;
                    }
                    let mut work = RecordReservation::from(context.reservation());
                    let [document_external_id, chunk_external_id, text] =
                        work.copy_strings(["doc", "chunk", "needle"], &context)?;
                    let row = ImportRow::Chunk {
                        document_external_id,
                        chunk: ChunkInput {
                            chunk_external_id,
                            text,
                            ordinal: if phase == "chunk" { 1 } else { 0 },
                            metadata: JsonValue::object(),
                            source_span: None,
                        },
                        line: 1,
                        source_span_metadata_conflict: false,
                    };
                    visits += 1;
                    if phase == "chunk" {
                        return crate::test_alloc::fail_byte_allocation_once(64, || {
                            emit(row, work)
                        });
                    }
                    emit(row, work)?;
                    let mut work = RecordReservation::from(context.reservation());
                    let [chunk_external_id, model_name] =
                        work.copy_strings(["chunk", "m"], &context)?;
                    let mut vector_work = context.workspace_reservation();
                    vector_work.grow_to(8, 8, "operation_scratch")?;
                    let vector = vec![1.0, 0.0];
                    work.merge_workspace(vector_work);
                    let row = ImportRow::Embedding {
                        input: EmbeddingInput {
                            chunk_external_id,
                            model_name,
                            vector,
                            dim: 2,
                        },
                        line: 1,
                    };
                    visits += 1;
                    crate::test_alloc::fail_byte_allocation_once(id_bytes, || emit(row, work))
                },
            );
            assert_eq!(result.unwrap_err().code(), ErrorCode::IoError);
            assert_eq!(
                visits,
                match phase {
                    "document" => 1,
                    "chunk" => 3,
                    _ => 3,
                }
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(db.temporary_disk_budget().used(), 0);
            assert!(db.collection("notes").is_err());
            let (values, work) = warnings.into_parts();
            assert!(values.is_empty());
            drop((values, work));
            assert_eq!(
                (
                    context.available_workspace_bytes(),
                    context.available_catalog_bytes(),
                    context.available_total_bytes()
                ),
                available
            );
            drop(db);
            let reader = Database::open_read_only(&path).unwrap();
            assert!(reader.collection("notes").is_err());
            drop(reader);
            fs::remove_file(path).unwrap();
        }
    }

    fn long_reference_inputs(directory: &Path) -> (String, String, [String; 3]) {
        fs::create_dir(directory).unwrap();
        fs::write(directory.join("manifest.json"), r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"embedding_model":"m","distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl"}"#).unwrap();
        let document = "d".repeat(600 * 1024);
        let chunk = "c".repeat(600 * 1024);
        let inputs = [
            format!("{{\"external_id\":\"{document}\",\"metadata\":{{}}}}\n"),
            format!("{{\"document_external_id\":\"{document}\",\"chunk_external_id\":\"{chunk}\",\"ordinal\":0,\"text\":\"needle\",\"metadata\":{{}}}}\n"),
            format!("{{\"chunk_external_id\":\"{chunk}\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}}\n"),
        ];
        for (name, text) in ["documents.jsonl", "chunks.jsonl", "embeddings.jsonl"]
            .iter()
            .zip(&inputs)
        {
            fs::write(directory.join(name), text).unwrap();
        }
        (document, chunk, inputs)
    }

    #[test]
    fn manifest_single_long_reference_keys_use_working_memory_not_cache_batch_target() {
        for cache in [0, 1] {
            let directory = std::env::temp_dir().join(format!(
                "cofferdb-long-references-{cache}-{}",
                std::process::id()
            ));
            let (document_id, chunk_id, _) = long_reference_inputs(&directory);
            let path = directory.join("data.coffer");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    resource_profile: ResourceProfile::Constrained,
                    cache_budget_mb: Some(cache),
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            let context = db.json_context();
            let summary = db
                .import_manifest(directory.join("manifest.json"), OnConflict::Error)
                .unwrap();
            assert_eq!(
                (
                    summary.documents_inserted,
                    summary.chunks_inserted,
                    summary.embeddings_inserted
                ),
                (1, 1, 1)
            );
            assert_eq!(db.temporary_disk_budget().used(), 0);
            let collection = db.collection("notes").unwrap();
            let chunk = collection.get_chunk_by_external_id(&chunk_id).unwrap();
            assert_eq!(chunk.chunk_external_id, chunk_id);
            assert_eq!(
                collection
                    .get_document(&chunk.document_id)
                    .unwrap()
                    .external_id,
                document_id
            );
            assert!(chunk.embedding_id.is_some());
            let query = collection
                .search()
                .text("needle")
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                });
            let expected = query.execute().unwrap();
            assert_eq!(expected.results.len(), 1);
            assert_eq!(expected.results[0].chunk_external_id, chunk_id);
            assert_eq!(expected.results[0].vector_score, Some(1.0));
            db.close().unwrap();
            assert_eq!(
                context.available_workspace_bytes(),
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
            );
            assert_eq!(
                context.available_total_bytes(),
                JsonOperationContext::standalone().available_total_bytes()
            );
            let reopened = Database::open_read_only(&path).unwrap();
            let actual = reopened
                .collection("notes")
                .unwrap()
                .search()
                .text("needle")
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .execute()
                .unwrap();
            assert_eq!(actual.results.len(), 1);
            assert_eq!(actual.results[0].chunk_external_id, chunk_id);
            assert_eq!(actual.results[0].vector_score, Some(1.0));
            assert_eq!(
                actual.results[0].final_score.to_bits(),
                expected.results[0].final_score.to_bits()
            );
            reopened.close().unwrap();
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn manifest_duplicate_long_reference_keys_fail_before_any_commit() {
        let names = ["documents.jsonl", "chunks.jsonl", "embeddings.jsonl"];
        for (duplicate, name) in names.iter().enumerate() {
            let directory = std::env::temp_dir().join(format!(
                "cofferdb-long-reference-duplicate-{duplicate}-{}",
                std::process::id()
            ));
            let (_, _, inputs) = long_reference_inputs(&directory);
            let repeated = if duplicate == 1 {
                inputs[duplicate].replace("\"ordinal\":0", "\"ordinal\":1")
            } else {
                inputs[duplicate].clone()
            };
            fs::write(
                directory.join(name),
                format!("{}\n{repeated}", inputs[duplicate]),
            )
            .unwrap();
            let path = directory.join("data.coffer");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    resource_profile: ResourceProfile::Constrained,
                    cache_budget_mb: Some(0),
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            let context = db.json_context();
            let available = context.available_workspace_bytes();
            let available_total = context.available_total_bytes();
            let before = fs::read(&path).unwrap();
            let error = db
                .import_manifest(directory.join("manifest.json"), OnConflict::Error)
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::DuplicateExternalId, "{error:?}");
            for (key, expected) in [
                ("path", directory.join(name).display().to_string()),
                ("line", "3".into()),
                ("first_line", "1".into()),
            ] {
                assert!(
                    error
                        .detail()
                        .iter()
                        .any(|(actual_key, value)| actual_key == key && value == &expected),
                    "{error:?}"
                );
            }
            assert_eq!(fs::read(&path).unwrap(), before);
            assert!(db.list_collections().unwrap().is_empty());
            assert_eq!(db.temporary_disk_budget().used(), 0);
            assert_eq!(context.available_workspace_bytes(), available);
            assert_eq!(context.available_total_bytes(), available_total);
            db.close().unwrap();
            fs::remove_dir_all(directory).unwrap();
        }
    }

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
        let before = fs::read(&path).unwrap();
        assert_eq!(
            db.import_manifest(&manifest, OnConflict::Skip)
                .unwrap_err()
                .code(),
            ErrorCode::ChunkNotFound
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        let embeddings = fs::read_to_string(directory.join("embeddings.jsonl")).unwrap();
        let retained = embeddings
            .lines()
            .filter(|line| {
                let value = crate::json::parse(line).unwrap();
                let id: usize = value
                    .get("chunk_external_id")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .strip_prefix("c-")
                    .unwrap()
                    .parse()
                    .unwrap();
                !(id / 2 % 2 == 0 && id % 2 == 1)
            })
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        fs::write(directory.join("embeddings.jsonl"), retained).unwrap();
        let summary = db.import_manifest(&manifest, OnConflict::Skip).unwrap();
        assert_eq!(
            (summary.documents_inserted, summary.documents_skipped),
            (4, 4)
        );
        assert_eq!((summary.chunks_inserted, summary.chunks_skipped), (8, 8));
        assert_eq!(
            (summary.embeddings_inserted, summary.embeddings_skipped),
            (8, 4)
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
