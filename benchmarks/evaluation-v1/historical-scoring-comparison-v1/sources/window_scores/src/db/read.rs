use super::Database;
use crate::catalog::{VisibleChunk, VisibleEmbedding};
use crate::json::{JsonBuilder, JsonInputKind, JsonOperationContext, JsonReservation, JsonValue};
use crate::storage::PayloadReader;
use crate::Result;

pub(super) fn read_visible_chunk_text(
    reader: &mut PayloadReader,
    visible: &VisibleChunk<'_>,
    context: &JsonOperationContext,
    bytes_read: &mut u64,
) -> Result<crate::json::ReservedString> {
    let Some(record_ref) = visible.record_ref else {
        return crate::json::ReservedString::copy(&visible.chunk.text, context);
    };
    *bytes_read = bytes_read.saturating_add(record_ref.payload.len);
    reader.read_chunk_text_in(record_ref, context)
}

pub(super) fn read_visible_embedding_vector(
    reader: &mut PayloadReader,
    visible: &VisibleEmbedding<'_>,
    context: &JsonOperationContext,
    bytes_read: &mut u64,
) -> Result<crate::json::ReservedVec<f32>> {
    let Some(record_ref) = visible.record_ref else {
        let mut vector = crate::json::ReservedVec::with_capacity(
            visible.embedding.vector.len(),
            context,
            "operation_scratch",
        )?;
        for value in &visible.embedding.vector {
            vector.push(*value);
        }
        return Ok(vector);
    };
    *bytes_read = bytes_read.saturating_add(record_ref.payload.len);
    reader.read_embedding_vector_in(record_ref, context)
}

pub(crate) enum RecordKey<'a> {
    Id(&'a str),
    ExternalId(&'a str),
}

impl Database {
    pub(crate) fn document_json_in_collection(
        &self,
        collection_id: &str,
        key: RecordKey<'_>,
        include_chunks: bool,
        context: &JsonOperationContext,
    ) -> Result<(JsonValue, JsonReservation)> {
        let inner = self.lock()?;
        let document = match key {
            RecordKey::Id(id) => inner.catalog.document_by_id_ref(collection_id, id)?,
            RecordKey::ExternalId(id) => {
                inner.catalog.document_by_external_id(collection_id, id)?
            }
        };
        let (document_json, mut work) = document.to_json_in(context)?;
        let (chunks_count, chunks) = if include_chunks {
            let summaries = inner.catalog.chunks_for_document_in(
                collection_id,
                &document.document_id,
                context,
            )?;
            let chunks_count = summaries.len();
            let mut chunks = Vec::new();
            work.reserve_vec(&mut chunks, chunks_count, JsonInputKind::StoredJson)?;
            for summary in summaries.iter() {
                let (chunk, mut chunk_work) =
                    inner.materialize_chunk_in(&summary.chunk_id, context)?;
                let value = chunk.into_json_retained(&mut chunk_work)?;
                work.merge(chunk_work.into_json());
                chunks.push(value);
            }
            (chunks_count, Some(JsonValue::Array(chunks)))
        } else {
            (
                inner
                    .catalog
                    .chunk_count_for_document(collection_id, &document.document_id)?,
                None,
            )
        };
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
        let mut result = builder.object([
            ("document", document_json),
            ("document_id", builder.string(&document.document_id)?),
            ("external_id", builder.string(&document.external_id)?),
            ("chunks_count", JsonValue::Number(chunks_count as f64)),
        ])?;
        if let Some(chunks) = chunks {
            builder.insert_if_absent(result.as_object_mut().unwrap(), "chunks", || Ok(chunks))?;
        }
        Ok((result, work))
    }

    pub(crate) fn chunk_json_in_collection(
        &self,
        collection_id: &str,
        key: RecordKey<'_>,
        context: &JsonOperationContext,
    ) -> Result<(JsonValue, JsonReservation)> {
        let inner = self.lock()?;
        let summary = match key {
            RecordKey::Id(id) => inner.catalog.chunk_by_id(collection_id, id)?,
            RecordKey::ExternalId(id) => inner.catalog.chunk_by_external_id(collection_id, id)?,
        };
        let (payload, mut work) = inner.materialize_chunk_in(&summary.chunk_id, context)?;
        let chunk = payload;
        let document = inner
            .catalog
            .document_by_id_ref(collection_id, &chunk.document_id)?;
        let (document_json, document_work) = document.to_json_in(context)?;
        work.merge(document_work);
        let embedding_present = chunk.embedding_id.is_some();
        let chunk_json = chunk.into_json_retained(&mut work)?;
        let mut work = work.into_json();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
        let result = builder.object([
            ("chunk", chunk_json),
            ("chunk_id", builder.string(&summary.chunk_id)?),
            (
                "chunk_external_id",
                builder.string(&summary.chunk_external_id)?,
            ),
            ("document", document_json),
            ("document_id", builder.string(&document.document_id)?),
            ("external_id", builder.string(&document.external_id)?),
            ("embedding_present", JsonValue::Bool(embedding_present)),
        ])?;
        Ok((result, work))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::text::TextSearchOptions;
    use crate::json::{object_from_pairs, JsonOutputSchema, JsonProcessingPolicyV1};
    use crate::{ChunkInput, CollectionConfig, DatabaseConfig, DocumentInput, Error, ErrorCode};

    fn fixture(name: &str, chunk_metadata: JsonValue) -> (Database, std::path::PathBuf, String) {
        let path =
            std::env::temp_dir().join(format!("cofferdb-{name}-{}.coffer", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: crate::DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: crate::VectorIndexConfig::Exact,
                tokenizer: Default::default(),
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: Some("title".into()),
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..4)
                .rev()
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("doc#{ordinal}"),
                    ordinal,
                    text: "payload".into(),
                    metadata: chunk_metadata.clone(),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
        (db, path, col.collection_id().to_owned())
    }

    #[test]
    fn snapshot_payload_reads_preserve_values_counters_and_error_kinds() {
        let (db, path, _) = fixture("snapshot-payload-read", JsonValue::object());
        let col = db.collection("notes").unwrap();
        col.put_embeddings(vec![crate::EmbeddingInput {
            chunk_external_id: "doc#0".into(),
            model_name: "model".into(),
            dim: 2,
            vector: vec![0.25, -0.5],
        }])
        .unwrap();
        let before = std::fs::read(&path).unwrap();
        let context = db.json_context();
        let capacity = context.available_workspace_bytes();
        let inner = db.lock().unwrap();
        let snapshot = inner.catalog.visible_snapshot();
        let chunk = snapshot.live_chunks().next().unwrap();
        let embedding = snapshot.live_embeddings().next().unwrap();
        let mut reader = inner.storage.payload_reader().unwrap();
        let mut bytes = 7;
        assert_eq!(
            read_visible_chunk_text(&mut reader, &chunk, &context, &mut bytes)
                .unwrap()
                .as_ref(),
            "payload"
        );
        assert_eq!(bytes, 7 + chunk.record_ref.unwrap().payload.len);
        assert_eq!(
            read_visible_embedding_vector(&mut reader, &embedding, &context, &mut bytes)
                .unwrap()
                .as_slice(),
            vec![0.25, -0.5]
        );
        assert_eq!(
            bytes,
            7 + chunk.record_ref.unwrap().payload.len + embedding.record_ref.unwrap().payload.len
        );
        let wrong_chunk = VisibleChunk {
            chunk: chunk.chunk,
            record_ref: embedding.record_ref,
        };
        assert_eq!(
            read_visible_chunk_text(&mut reader, &wrong_chunk, &context, &mut bytes)
                .err()
                .unwrap()
                .code(),
            ErrorCode::FileFormatError
        );
        let wrong_embedding = VisibleEmbedding {
            embedding: embedding.embedding,
            record_ref: chunk.record_ref,
        };
        assert_eq!(
            read_visible_embedding_vector(&mut reader, &wrong_embedding, &context, &mut bytes)
                .err()
                .unwrap()
                .code(),
            ErrorCode::FileFormatError
        );
        let mut owned_chunk = chunk.chunk.clone();
        owned_chunk.text = "owned text".into();
        let mut owned_embedding = embedding.embedding.clone();
        owned_embedding.vector = vec![1.0, 0.0];
        let previous_bytes = bytes;
        assert_eq!(
            read_visible_chunk_text(
                &mut reader,
                &VisibleChunk {
                    chunk: &owned_chunk,
                    record_ref: None
                },
                &context,
                &mut bytes,
            )
            .unwrap()
            .as_ref(),
            "owned text"
        );
        assert_eq!(
            read_visible_embedding_vector(
                &mut reader,
                &VisibleEmbedding {
                    embedding: &owned_embedding,
                    record_ref: None
                },
                &context,
                &mut bytes,
            )
            .unwrap()
            .as_slice(),
            vec![1.0, 0.0]
        );
        assert_eq!(bytes, previous_bytes);
        bytes = u64::MAX - 1;
        read_visible_chunk_text(&mut reader, &chunk, &context, &mut bytes).unwrap();
        read_visible_embedding_vector(&mut reader, &embedding, &context, &mut bytes).unwrap();
        assert_eq!(bytes, u64::MAX);

        let mut busy = context.reservation();
        busy.grow(capacity, JsonInputKind::StoredJson).unwrap();
        let error = read_visible_chunk_text(&mut reader, &chunk, &context, &mut bytes)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        for (key, value) in [
            ("resource", "memory".into()),
            ("operation", "operation_scratch".into()),
            (
                "limit_bytes",
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES.to_string(),
            ),
            (
                "requested_bytes",
                (JsonProcessingPolicyV1::WORK_BUDGET_BYTES + 20).to_string(),
            ),
        ] {
            assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
        }
        // Binary vectors have no JSON subtree, but their owned buffer needs shared workspace.
        let error = read_visible_embedding_vector(&mut reader, &embedding, &context, &mut bytes)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "operation" && value == "operation_scratch"));
        drop(busy);
        read_visible_chunk_text(&mut reader, &chunk, &context, &mut bytes).unwrap();
        read_visible_embedding_vector(&mut reader, &embedding, &context, &mut bytes).unwrap();
        context
            .reservation()
            .grow(capacity, JsonInputKind::StoredJson)
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop((reader, inner));
        drop((col, db));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn text_build_and_two_pass_search_hold_each_text_through_tokenization() {
        use crate::index::text::TextIndex;
        use crate::json::ReservedString;
        use std::{cell::Cell, rc::Rc};

        struct CheckedText {
            text: ReservedString,
            context: JsonOperationContext,
            available: usize,
            visits: Rc<Cell<usize>>,
        }
        impl AsRef<str> for CheckedText {
            fn as_ref(&self) -> &str {
                let text = self.text.as_ref();
                let rest = self.available - text.len();
                let mut work = self.context.workspace_reservation();
                work.grow_to(rest, rest, "operation_scratch").unwrap();
                assert!(work
                    .grow_to(rest + 1, rest + 1, "operation_scratch")
                    .is_err());
                self.visits.set(self.visits.get() + 1);
                text
            }
        }

        let (db, path, collection_id) = fixture("text-scan-workspace", JsonValue::object());
        let before = std::fs::read(&path).unwrap();
        let context = db.json_context();
        let inner = db.lock().unwrap();
        let snapshot = inner.catalog.visible_snapshot();
        let reference =
            TextIndex::try_build_matching(&snapshot, |_| true, |_| Ok("payload")).unwrap();
        let limit = context.available_workspace_bytes();
        for seed in 0..100usize {
            // Include the metadata-validation stack, text and live candidate allocations.
            let available = 32768 + seed;
            let mut held = context.workspace_reservation();
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            for build in [false, true] {
                let mut reader = inner.storage.payload_reader().unwrap();
                let visits = Rc::new(Cell::new(0));
                let mut bytes = 0;
                let mut loaded = 0;
                let loader = |visible: &VisibleChunk<'_>| {
                    let available = context
                        .workspace_reservation()
                        .grow_to(0, available, "operation_scratch")
                        .unwrap();
                    loaded += 1;
                    if seed % 5 == 0 && loaded == 3 {
                        return Err(Error::new(ErrorCode::IoError, "injected text read failure"));
                    }
                    Ok(CheckedText {
                        text: read_visible_chunk_text(&mut reader, visible, &context, &mut bytes)?,
                        context: context.clone(),
                        available,
                        visits: visits.clone(),
                    })
                };
                let query = if seed % 3 == 0 { "missing" } else { "payload" };
                let result = if build {
                    TextIndex::try_build_matching(&snapshot, |_| true, loader).and_then(|index| {
                        index.search(
                            &collection_id,
                            query,
                            TextSearchOptions::with_explanations(4),
                            &context,
                        )
                    })
                } else {
                    TextIndex::search_from_snapshot(
                        &snapshot,
                        &collection_id,
                        query,
                        TextSearchOptions::with_explanations(4),
                        &context,
                        loader,
                    )
                };
                if seed % 5 == 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::IoError, "{error:?}");
                    assert_eq!(visits.get(), 2);
                } else {
                    let actual = result.unwrap();
                    let expected = reference
                        .search(
                            &collection_id,
                            query,
                            TextSearchOptions::with_explanations(4),
                            &JsonOperationContext::standalone(),
                        )
                        .unwrap();
                    assert_eq!(
                        actual
                            .iter()
                            .map(|hit| (&hit.chunk_id, hit.score.to_bits()))
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|hit| (&hit.chunk_id, hit.score.to_bits()))
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        visits.get(),
                        if build || query == "missing" { 4 } else { 8 }
                    );
                }
                context
                    .workspace_reservation()
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
            }
        }
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(inner);
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn embedding_dimension_diagnosis_keeps_payload_charged_and_borrows_summary_fallback() {
        let (db, path, _) = fixture("embedding-diagnosis-workspace", JsonValue::object());
        let col = db.collection("notes").unwrap();
        col.put_embeddings(vec![crate::EmbeddingInput {
            chunk_external_id: "doc#0".into(),
            model_name: "model".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        let before = std::fs::read(&path).unwrap();
        let context = db.json_context();
        let mut inner = db.lock().unwrap();
        let collection = inner.catalog.collection_by_name("notes").unwrap().clone();
        let chunk = inner
            .catalog
            .live_chunks()
            .find(|chunk| chunk.embedding_id.is_some())
            .unwrap()
            .clone();
        let id = chunk.embedding_id.as_ref().unwrap();
        let record_ref = inner.catalog.record_ref_for_embedding(id).unwrap().clone();
        let limit = context.available_workspace_bytes();
        for seed in 0..100usize {
            let mut held = context.workspace_reservation();
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            let error = super::super::expected_embedding_dim_diagnosis(
                &inner,
                &collection,
                Some(&chunk),
                &context,
            )
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            held.retain(limit - 1024);
            let actual = super::super::expected_embedding_dim_diagnosis(
                &inner,
                &collection,
                Some(&chunk),
                &context,
            )
            .unwrap();
            assert!(actual.matches);
            assert_eq!(
                (actual.actual_dim, actual.actual_vector_len),
                (Some(2), Some(2))
            );
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            inner.catalog.record_refs.remove(id);
            let summary = inner.catalog.embeddings.get_mut(id).unwrap();
            summary.dim = 2;
            summary.vector = vec![1.0; 2 + seed % 2];
            let actual = super::super::expected_embedding_dim_diagnosis(
                &inner,
                &collection,
                Some(&chunk),
                &context,
            )
            .unwrap();
            assert_eq!(actual.matches, seed % 2 == 0);
            assert_eq!(actual.actual_vector_len, Some(2 + seed % 2));
            inner
                .catalog
                .record_refs
                .insert(id.clone(), record_ref.clone());
        }
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(inner);
        drop((col, db));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn exact_vector_scan_keeps_one_buffer_live_and_releases_every_exit_path() {
        use crate::index::vector::{ExactVectorIndex, VectorSearchOptions};

        let (db, path, collection_id) = fixture("bounded-vector-scan", JsonValue::object());
        let col = db.collection("notes").unwrap();
        col.put_embeddings(
            (0..4)
                .map(|index| crate::EmbeddingInput {
                    chunk_external_id: format!("doc#{index}"),
                    model_name: "model".into(),
                    dim: 2,
                    vector: vec![(index + 1) as f32, 0.0],
                })
                .collect(),
        )
        .unwrap();
        let before = std::fs::read(&path).unwrap();
        let inner = db.lock().unwrap();
        let snapshot = inner.catalog.visible_snapshot();
        let minimum = snapshot
            .live_embeddings()
            .map(|visible| {
                let value = visible.embedding;
                value.embedding_id.len()
                    + value.chunk_id.len()
                    + value.collection_id.len()
                    + value.model_name.len()
                    + value.created_at.len()
                    + 2 * std::mem::size_of::<f32>()
            })
            .max()
            .unwrap();
        let limit = db.json_context().available_workspace_bytes();
        let candidate_ids = snapshot
            .live_embeddings()
            .map(|visible| visible.embedding.chunk_id.len())
            .collect::<Vec<_>>();
        let candidate_array = 4 * std::mem::size_of::<crate::index::vector::VectorHit>();
        for seed in 0..100usize {
            let context = db.json_context();
            let available =
                minimum + candidate_array + candidate_ids.iter().sum::<usize>() + seed % 3;
            let mut held = context.workspace_reservation();
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            let mut reader = inner.storage.payload_reader().unwrap();
            let mut loaded = 0;
            let mut bytes = 0;
            let query: &[f32] = if seed % 3 == 0 {
                &[1.0, 0.0, 0.0]
            } else {
                &[1.0, 0.0]
            };
            let result = ExactVectorIndex.search_with_vectors(
                &snapshot,
                &collection_id,
                query,
                VectorSearchOptions {
                    metric: crate::DistanceMetric::DotProduct,
                    vectors_normalized: false,
                    limit: 4,
                },
                |_| true,
                &context,
                |visible| {
                    let retained = if loaded == 0 || query.len() == 3 {
                        0
                    } else {
                        candidate_array + candidate_ids[..loaded].iter().sum::<usize>()
                    };
                    let remaining = available - retained;
                    context
                        .workspace_reservation()
                        .grow_to(remaining, remaining, "operation_scratch")
                        .unwrap();
                    loaded += 1;
                    if seed % 5 == 0 && loaded == 3 {
                        return Err(Error::new(
                            ErrorCode::IoError,
                            "injected vector read failure",
                        ));
                    }
                    let vector =
                        read_visible_embedding_vector(&mut reader, visible, &context, &mut bytes)?;
                    let mut rest = context.workspace_reservation();
                    rest.grow_to(remaining - 8, remaining - 8, "operation_scratch")
                        .unwrap();
                    assert!(rest
                        .grow_to(remaining - 7, remaining - 7, "operation_scratch")
                        .is_err());
                    Ok(vector)
                },
            );
            if seed % 5 == 0 {
                assert_eq!(result.unwrap_err().code(), ErrorCode::IoError);
                assert_eq!(loaded, 3);
            } else {
                let (hits, count) = result.unwrap();
                assert_eq!(loaded, 4);
                if query.len() == 3 {
                    assert!(hits.is_empty());
                    assert_eq!(count, 0);
                } else {
                    assert_eq!(count, 4);
                    assert_eq!(
                        hits.iter().map(|hit| hit.raw_score).collect::<Vec<_>>(),
                        vec![4.0, 3.0, 2.0, 1.0]
                    );
                }
            }
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(inner);
        drop((col, db));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn document_count_avoids_payload_reads_and_combined_read_retains_budget() {
        let metadata = object_from_pairs([("payload", JsonValue::String("x".repeat(1024 * 1024)))]);
        let (db, path, collection_id) = fixture("read-dto-budget", metadata);
        let before = std::fs::read(&path).unwrap();
        let capacity = db.json_context().available_workspace_bytes();
        let mut busy = db.json_context().reservation();
        busy.grow(capacity - 3 * 1024 * 1024, JsonInputKind::StoredJson)
            .unwrap();
        let context = db.json_context();
        let (value, mut work) = db
            .document_json_in_collection(
                &collection_id,
                RecordKey::ExternalId("doc"),
                false,
                &context,
            )
            .unwrap();
        assert_eq!(value.get("chunks_count"), Some(&JsonValue::Number(4.0)));
        assert!(value.get("chunks").is_none());
        value
            .pretty_retained_in(
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
                &context,
                &mut work,
            )
            .unwrap();
        drop((value, work));
        let error = db
            .document_json_in_collection(
                &collection_id,
                RecordKey::ExternalId("doc"),
                true,
                &context,
            )
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        for (key, expected) in [
            ("json_input_kind", "stored_json"),
            ("json_limit_reason", "work_budget_exceeded"),
        ] {
            assert!(error
                .detail()
                .iter()
                .any(|(name, value)| name == key && value == expected));
        }
        busy.grow(3 * 1024 * 1024, JsonInputKind::StoredJson)
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(busy);
        let (value, work) = db
            .document_json_in_collection(
                &collection_id,
                RecordKey::ExternalId("doc"),
                true,
                &context,
            )
            .unwrap();
        let chunks = value.get("chunks").unwrap().as_array().unwrap();
        for (ordinal, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.get("ordinal").unwrap().as_f64(), Some(ordinal as f64));
        }
        let measured = value
            .reserve_retained_in(JsonInputKind::StoredJson, &context)
            .unwrap();
        assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
        drop((value, work, measured));
        context
            .reservation()
            .grow(capacity, JsonInputKind::StoredJson)
            .unwrap();
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_dtos_preserve_lookup_visibility_and_public_fields() {
        let (db, path, collection_id) = fixture("read-dto-contract", JsonValue::object());
        let col = db.collection("notes").unwrap();
        let document = col.get_document_by_external_id("doc").unwrap();
        let chunks = col.list_chunks(&document.document_id).unwrap();
        let expected = object_from_pairs([
            ("document", document.to_json()),
            (
                "document_id",
                JsonValue::String(document.document_id.clone()),
            ),
            (
                "external_id",
                JsonValue::String(document.external_id.clone()),
            ),
            ("chunks_count", JsonValue::Number(4.0)),
            (
                "chunks",
                JsonValue::Array(chunks.iter().map(|chunk| chunk.to_json()).collect()),
            ),
        ]);
        let context = db.json_context();
        for key in [
            RecordKey::Id(&document.document_id),
            RecordKey::ExternalId("doc"),
        ] {
            let (value, work) = db
                .document_json_in_collection(&collection_id, key, true, &context)
                .unwrap();
            assert_eq!(value, expected);
            drop((value, work));
        }
        let chunk = &chunks[0];
        let expected = object_from_pairs([
            ("chunk", chunk.to_json()),
            ("chunk_id", JsonValue::String(chunk.chunk_id.clone())),
            (
                "chunk_external_id",
                JsonValue::String(chunk.chunk_external_id.clone()),
            ),
            ("document", document.to_json()),
            (
                "document_id",
                JsonValue::String(document.document_id.clone()),
            ),
            (
                "external_id",
                JsonValue::String(document.external_id.clone()),
            ),
            ("embedding_present", JsonValue::Bool(false)),
        ]);
        for key in [
            RecordKey::Id(&chunk.chunk_id),
            RecordKey::ExternalId(&chunk.chunk_external_id),
        ] {
            let (value, work) = db
                .chunk_json_in_collection(&collection_id, key, &context)
                .unwrap();
            assert_eq!(value, expected);
            let measured = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
        }
        col.delete_chunk(&chunk.chunk_id).unwrap();
        let (value, _) = db
            .document_json_in_collection(
                &collection_id,
                RecordKey::Id(&document.document_id),
                false,
                &context,
            )
            .unwrap();
        assert_eq!(value.get("chunks_count"), Some(&JsonValue::Number(3.0)));
        assert_eq!(
            db.chunk_json_in_collection(&collection_id, RecordKey::Id(&chunk.chunk_id), &context)
                .err()
                .unwrap()
                .code(),
            ErrorCode::ChunkNotFound
        );
        col.delete_document(&document.document_id).unwrap();
        assert_eq!(
            db.document_json_in_collection(
                &collection_id,
                RecordKey::ExternalId("doc"),
                false,
                &context
            )
            .err()
            .unwrap()
            .code(),
            ErrorCode::DocumentNotFound
        );
        drop((col, db));
        std::fs::remove_file(path).unwrap();
    }
}
