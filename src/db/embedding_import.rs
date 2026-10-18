use super::*;
use crate::import::{embedding_from_json_in, with_jsonl_location};
use crate::json::{JsonLines, JsonSchema, ReservedString};
use crate::storage::PreparedRecordsBuilder;
use crate::temporary::ExternalUniqueKeys;

impl Database {
    pub(crate) fn import_embeddings_jsonl(
        &self,
        collection_id: &str,
        path: &Path,
        on_conflict: OnConflict,
    ) -> Result<InsertSummary> {
        let context = self.json_context();
        let mut inner = self.lock()?;
        let mut collection_work = RecordReservation::from(context.reservation());
        let (collection, work) = inner.read_reclaiming(|inner| {
            copy_collection_in(inner.catalog.collection_by_id(collection_id)?, &context)
        })?;
        collection_work.merge(work);
        let mut models = inner
            .read_reclaiming(|inner| ModelAdmission::new(&inner.catalog, &collection, &context))?;
        let mut preflight_error = None;
        let input = File::open(path)
            .map_err(|error| Error::from(error).with_detail("path", path.display()))?;
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
                disk.clone(),
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
        let reference_workspace = (inner.active_cache_budget_mb().saturating_mul(1_048_576) / 128)
            .clamp(64 * 1024, 1_048_576);
        let mut keys = inner.read_reclaiming(|_| {
            ExternalUniqueKeys::new_growing(
                &std::env::temp_dir(),
                disk.clone(),
                reference_workspace,
                &context,
                ErrorCode::DuplicateExternalId,
                "embedding for chunk appears more than once in the same import",
            )
        })?;
        let millis = time::unix_millis_now();
        let timestamp = inner.read_reclaiming(|_| {
            ReservedString::build(&context, "batch_prepare", |out| {
                time::write_unix_millis(out, millis)
            })
        })?;
        let mut summary = InsertSummary::default();
        let mut rows =
            JsonLines::new_reclaiming(input, JsonSchema::EmbeddingRow, &context, u64::MAX, || {
                reclaim_caches(&mut inner)
            })
            .map_err(|error| error.with_detail("path", path.display()))?;
        while let Some(row) = rows.next_reclaiming(|| reclaim_caches(&mut inner)) {
            let (line, parsed) = row.map_err(|error| error.with_detail("path", path.display()))?;
            let mut row_work = context.reservation();
            let (mut value, work) = parsed.into_parts();
            row_work.merge(work);
            let (input, conversion) = inner
                .read_reclaiming(|_| embedding_from_json_in(&mut value, &context))
                .map_err(|error| with_jsonl_location(error, path, line))?;
            row_work.merge(conversion);
            drop(value);
            row_work.retain(input.workspace_bytes());
            input
                .validate_structure()
                .map_err(|error| with_jsonl_location(error, path, line))?;
            keys.push_value_reclaiming(
                &[input.chunk_external_id.as_bytes()],
                line as u64,
                0,
                || reclaim_caches(&mut inner),
            )
            .map_err(|error| with_jsonl_location(error, path, line))?;
            let consumed = (|| {
                if preflight_error.is_some() {
                    return Ok(());
                }
                input.validate_target_dimension(collection.embedding_dim)?;
                let chunk = inner
                    .catalog
                    .chunk_by_external_id(collection_id, &input.chunk_external_id)?;
                if inner.catalog.embedding_for_chunk(&chunk.chunk_id).is_some() {
                    match on_conflict {
                        OnConflict::Error => {
                            return Err(Error::new(
                                ErrorCode::DuplicateExternalId,
                                format!(
                                    "embedding for chunk {} already exists",
                                    input.chunk_external_id
                                ),
                            ))
                        }
                        OnConflict::Skip => {
                            summary.embeddings_skipped += 1;
                            return Ok(());
                        }
                        OnConflict::Replace => summary.embeddings_replaced += 1,
                    }
                }
                match inner
                    .read_reclaiming(|_| models.validate_write(&input.model_name, None, &context))
                {
                    Err(error) if error.code() == ErrorCode::EmbeddingModelMismatch => {
                        preflight_error = Some(with_jsonl_location(error, path, line));
                        return Ok(());
                    }
                    result => result?,
                }
                let ([embedding_id, chunk_id, collection_id, created_at], _fields_work) = inner
                    .read_reclaiming(|inner| {
                        let mut work = RecordReservation::from(context.reservation());
                        let chunk = inner
                            .catalog
                            .chunk_by_external_id(collection_id, &input.chunk_external_id)?;
                        let (embedding_id, id_work) = inner
                            .catalog
                            .next_id_in("emb_", summary.embeddings_inserted as u64, &context)?
                            .into_parts();
                        work.merge_workspace(id_work);
                        let [chunk_id, collection_id, created_at] = work.copy_strings(
                            [&chunk.chunk_id, collection_id, timestamp.as_ref()],
                            &context,
                        )?;
                        Ok(([embedding_id, chunk_id, collection_id, created_at], work))
                    })?;
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
                    &mut row_work,
                )?;
                summary.embeddings_inserted += 1;
                Ok(())
            })();
            match consumed {
                Err(error)
                    if matches!(
                        error.code(),
                        ErrorCode::ChunkNotFound
                            | ErrorCode::DuplicateExternalId
                            | ErrorCode::InvalidEmbeddingDimension
                    ) =>
                {
                    preflight_error = Some(with_jsonl_location(error, path, line));
                }
                result => result.map_err(|error| with_jsonl_location(error, path, line))?,
            }
        }
        drop(rows);
        // Consume and validate the full input against one locked snapshot before any commit.
        keys.finish_reclaiming(|| reclaim_caches(&mut inner))
            .map_err(|error| error.with_detail("path", path.display()))?;
        if let Some(error) = preflight_error {
            return Err(error);
        }
        inner.append_import_batches(
            records.finish()?,
            ImportIndexBuildMode::DeferredFullCheckpoint,
            &context,
        )?;
        Ok(summary)
    }
}

fn reclaim_caches(inner: &mut Inner) -> bool {
    Inner::release_optional_caches(
        &mut inner.text_index,
        &mut inner.persistent_vector_segments,
        &mut inner.lazy_vector_segment,
        std::iter::empty(),
    )
}
