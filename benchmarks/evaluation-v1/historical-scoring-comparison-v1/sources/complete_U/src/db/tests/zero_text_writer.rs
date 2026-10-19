use super::*;

fn options() -> DatabaseOpenOptions {
    DatabaseOpenOptions {
        cache_budget_mb: Some(0),
        durability: Some(Durability::Fast),
        ..Default::default()
    }
}

fn input(id: &str, text: &str) -> DocumentInput {
    DocumentInput {
        external_id: id.into(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: vec![ChunkInput {
            chunk_external_id: id.into(),
            text: text.into(),
            ordinal: 0,
            metadata: JsonValue::object(),
            source_span: None,
        }],
    }
}

fn pressure(
    _inner: &Inner,
    context: &crate::json::JsonOperationContext,
    available: usize,
    parent: bool,
) -> (
    crate::json::WorkspaceReservation,
    Option<crate::json::CacheReservation>,
) {
    let mut held = context.workspace_reservation();
    let parent_held = if parent {
        Some(
            context
                .cache_reservation(
                    (context.available_total_bytes() - available) as u128,
                    usize::MAX,
                )
                .unwrap(),
        )
    } else {
        let occupied = context.available_workspace_bytes() - available;
        held.grow_to(occupied, occupied, "operation_scratch")
            .unwrap();
        None
    };
    assert_eq!(context.available_workspace_bytes(), available);
    (held, parent_held)
}

fn assert_checkpoint(db: &Database) {
    assert_checkpoint_with_retention(db, 0);
}

fn assert_checkpoint_with_retention(db: &Database, retained: u64) {
    let inner = db.lock().unwrap();
    if inner.lazy_text {
        assert!(inner.text_index.postings().is_empty());
        assert!(inner.text_index.chunk_lengths.is_empty());
        assert!(inner.text_index.total_chunks_by_collection.is_empty());
        assert!(inner.text_index.avg_len_by_collection.is_empty());
    }
    assert_eq!(
        inner.index_suffix_status,
        IndexSuffixState::PersistentCheckpoint
    );
    assert_eq!(
        inner.last_index_commit_sequence_no,
        Some(inner.last_commit_sequence_no)
    );
    if inner.lazy_text {
        assert!(inner.lazy_vector_segment.is_none());
        assert!(inner
            .persistent_vector_segments
            .as_ref()
            .map_or(true, |cache| cache.is_empty()));
    }
    let full = build_text_index(
        &inner.storage,
        &inner.catalog,
        &db.temporary_work,
        &db.json_context(),
    )
    .unwrap();
    if inner.text_index.has_resident_cache() {
        assert_eq!(inner.text_index.chunk_lengths, full.chunk_lengths);
        assert_eq!(
            inner
                .text_index
                .collection_counts()
                .into_iter()
                .filter(|(_, count)| *count != 0)
                .collect::<BTreeMap<_, _>>(),
            full.collection_counts()
        );
        assert_eq!(
            inner.text_index.avg_len_by_collection,
            full.avg_len_by_collection
        );
        assert_eq!(
            format!("{:?}", inner.text_index.postings()),
            format!("{:?}", full.postings())
        );
    }
    let source = inner.lazy_vector_source.as_ref().unwrap();
    for collection in inner.catalog.live_collections() {
        for query in [
            "shared alphaBeta",
            "needle storage_path",
            "absent",
            "the and",
        ] {
            let actual = TextIndex::search_from_segment(
                source,
                &inner.storage,
                &collection.collection_id,
                query,
                TextSearchOptions::with_explanations(30),
                &db.json_context(),
            )
            .unwrap()
            .unwrap();
            let expected = full
                .search(
                    &collection.collection_id,
                    query,
                    TextSearchOptions::with_explanations(30),
                    &db.json_context(),
                )
                .unwrap();
            assert_eq!(
                format!("{:?}", actual.as_slice()),
                format!("{:?}", expected.as_slice())
            );
        }
        assert_eq!(
            inner
                .text_stats()
                .unwrap()
                .term_count_in_collection(&collection.collection_id),
            full.terms_in_collection(&collection.collection_id).count()
        );
    }
    assert_eq!(inner.text_stats().unwrap().term_count(), full.term_count());
    let path = inner.path.clone();
    drop(inner);
    assert_eq!(db.temporary_work.used(), retained);
    assert!(scan_file(&path).unwrap().findings.is_empty());
}

#[test]
fn prepared_resident_build_does_not_tokenize_and_yields_read_buffer_on_both_axes() {
    let path = query_preflight_fixture("resident-prepared-budget", "shared needle");
    let db = Database::open_with_options(
        &path,
        DatabaseOpenOptions {
            cache_budget_mb: Some(1),
            ..options()
        },
    )
    .unwrap();
    let context = db.json_context();
    let inner = db.lock().unwrap();
    let text = inner
        .prepare_text_checkpoint(None, &BTreeSet::new(), &context)
        .unwrap();
    let before = fs::read(&path).unwrap();
    let disk = db.temporary_work.used();
    let minimum = 4 * 2 * std::mem::size_of::<u64>() + "shared".len();
    for parent in [false, true] {
        for allowance in [minimum - 1, minimum, minimum + 1] {
            let (held, parent_held) = pressure(&inner, &context, allowance, parent);
            crate::tokenizer::TOKEN_VISITS.with(|count| count.set(0));
            TEXT_BUILDS.with(|count| count.set(0));
            let result = build_text_index_from_checkpoint(
                text.as_ref(),
                &inner.catalog,
                |_| true,
                &context,
                1_048_576,
            );
            assert_eq!(crate::tokenizer::TOKEN_VISITS.with(Cell::get), 0);
            assert_eq!(TEXT_BUILDS.with(Cell::get), 0);
            if allowance < minimum {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                let limit = if parent {
                    3 * 1024 * 1024 * 1024usize
                } else {
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                };
                for (key, expected) in [
                    ("resource", "memory".to_owned()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", (limit + 1).to_string()),
                ] {
                    assert_eq!(error_detail_value(&error, key), Some(expected.as_str()));
                }
                assert_eq!(
                    error_detail_value(&error, "operation"),
                    Some("operation_scratch")
                );
            } else if parent {
                let index = result.unwrap();
                assert!(!index.has_resident_cache());
                assert!(index.chunk_lengths.is_empty());
            } else {
                let index = result.unwrap();
                assert_eq!(index.chunk_lengths.len(), 6);
                assert_eq!(index.term_count(), 2);
                assert_eq!(index.total_chunks_by_collection.len(), 2);
                assert_eq!(
                    index
                        .avg_len_by_collection
                        .values()
                        .copied()
                        .collect::<Vec<_>>(),
                    vec![2.0, 2.0]
                );
            }
            assert_eq!(context.available_workspace_bytes(), allowance);
            assert_eq!(db.temporary_work.used(), disk);
            assert_eq!(fs::read(&path).unwrap(), before);
            drop((held, parent_held));
        }
    }
    drop(text);
    assert_eq!(db.temporary_work.used(), 0);
    drop(inner);
    assert_checkpoint(&db);
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn first_compact_of_empty_database_prepares_a_complete_empty_text_source() {
    for cache in [0, 1] {
        let path = temp_path("empty-first-compact-source");
        let db = Database::create(
            &path,
            DatabaseConfig {
                cache_budget_mb: Some(cache),
                durability: Durability::Fast,
                ..Default::default()
            },
        )
        .unwrap();
        let pinned = Database::open_read_only(&path).unwrap();
        let old_bytes = fs::metadata(&path).unwrap().len();
        let summary = db
            .compact(CompactOptions {
                rebuild_indexes: true,
            })
            .unwrap();
        assert_eq!(summary.index_rebuild, "full_checkpoint_written");
        assert_eq!(db.stats().unwrap().document_count, 0);
        assert_eq!(pinned.stats().unwrap().last_commit_id, 0);
        assert_eq!(pinned.stats().unwrap().document_count, 0);
        assert_eq!(
            db.lock().unwrap().index_suffix_status,
            IndexSuffixState::MetadataRecordOnly
        );
        assert!(db.lock().unwrap().text_index.chunk_lengths.is_empty());
        assert_eq!(db.temporary_work.used(), old_bytes);
        assert!(scan_file(&path).unwrap().findings.is_empty());
        drop(pinned);
        assert_eq!(db.temporary_work.used(), 0);
        drop(db);
        let reopened = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(cache),
                ..options()
            },
        )
        .unwrap();
        assert_eq!(
            reopened.lock().unwrap().index_suffix_status,
            IndexSuffixState::MetadataRecordOnly
        );
        assert!(reopened.lock().unwrap().text_index.chunk_lengths.is_empty());
        drop(reopened);
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn zero_cache_writer_create_mutations_rebuild_compact_and_reopen_keep_postings_absent() {
    for profile in [
        ResourceProfile::QualityFirst,
        ResourceProfile::Balanced,
        ResourceProfile::ResourceFirst,
    ] {
        let path = temp_path("zero-writer-lifecycle");
        let db = Database::create(
            &path,
            DatabaseConfig {
                cache_budget_mb: Some(0),
                resource_profile: profile,
                durability: Durability::Fast,
                ..Default::default()
            },
        )
        .unwrap();
        for owner in ["first", "second", "empty"] {
            db.create_collection(CollectionConfig {
                name: owner.into(),
                embedding_dim: 2,
                tokenizer: Default::default(),
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
            assert_checkpoint(&db);
        }
        let first = db.collection("first").unwrap();
        let second = db.collection("second").unwrap();
        first
            .insert_document(input("one", "shared alphaBeta needle needle"))
            .unwrap();
        second
            .insert_document(input("two", "shared storage_path \u{e9}"))
            .unwrap();
        assert_checkpoint(&db);
        first
            .put_embeddings(vec![EmbeddingInput {
                chunk_external_id: "one".into(),
                model_name: "m".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            }])
            .unwrap();
        assert_checkpoint(&db);
        let pinned = Database::open_read_only(&path).unwrap();
        let reference = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                resource_profile: Some(profile),
                ..options()
            },
        )
        .unwrap();
        let old = reference
            .collection("first")
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
        let actual = first
            .search()
            .text("needle")
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .execute()
            .unwrap();
        assert_eq!(
            actual.results[0].explain.to_compact_string(),
            old.results[0].explain.to_compact_string()
        );
        assert_eq!(actual.query_context.index_source, "persistent_checkpoint");
        let chunk = first.get_chunk_by_external_id("one").unwrap();
        let why = db
            .why_not_debug(
                "first",
                &chunk.chunk_id,
                Some("needle".into()),
                None,
                JsonValue::object(),
                None,
                false,
                3,
                None,
            )
            .unwrap();
        let expected = reference
            .why_not_debug(
                "first",
                &chunk.chunk_id,
                Some("needle".into()),
                None,
                JsonValue::object(),
                None,
                false,
                3,
                None,
            )
            .unwrap();
        assert_eq!(
            why.result.to_compact_string(),
            expected.result.to_compact_string()
        );
        db.insert_document_in_collection(
            first.collection_id(),
            input("one", "new shared storage_path"),
            OnConflict::Replace,
        )
        .unwrap();
        assert_checkpoint(&db);
        let deleted = second.get_chunk_by_external_id("two").unwrap();
        second.delete_chunk(&deleted.chunk_id).unwrap();
        assert_checkpoint(&db);
        assert_eq!(
            db.rebuild_index(Some("first")).unwrap().rebuild_scope,
            "full_visible_snapshot"
        );
        assert_checkpoint(&db);
        let retained = fs::metadata(&path).unwrap().len();
        db.compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
        assert_checkpoint_with_retention(&db, retained);
        assert_eq!(
            pinned
                .collection("first")
                .unwrap()
                .search()
                .text("needle")
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        db.compact(CompactOptions {
            rebuild_indexes: false,
        })
        .unwrap();
        assert_eq!(
            db.lock().unwrap().index_suffix_status,
            IndexSuffixState::Missing
        );
        assert!(db.lock().unwrap().text_index.postings().is_empty());
        db.rebuild_index(None).unwrap();
        assert_checkpoint_with_retention(&db, retained);
        db.drop_collection("second").unwrap();
        assert_checkpoint_with_retention(&db, retained);
        drop(pinned);
        assert_eq!(db.temporary_work.used(), retained);
        drop(reference);
        assert_eq!(db.temporary_work.used(), 0);
        drop((first, second, db));
        let db = Database::open_with_options(&path, options()).unwrap();
        assert_checkpoint(&db);
        drop(db);
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn zero_cache_writer_preserves_untouched_text_before_owned_and_prepared_append_truncate() {
    for cache in [0, 1] {
        for prepared in [false, true] {
            let path = query_preflight_fixture("zero-writer-prefetch", "shared needle");
            let db = Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    cache_budget_mb: Some(cache),
                    ..options()
                },
            )
            .unwrap();
            assert_eq!(db.lock().unwrap().lazy_text, cache == 0);
            let offset = {
                let inner = db.lock().unwrap();
                let chunk = inner
                    .catalog
                    .chunk_by_external_id(
                        &inner
                            .catalog
                            .collection_by_name("second")
                            .unwrap()
                            .collection_id,
                        "chunk-1",
                    )
                    .unwrap();
                inner
                    .catalog
                    .record_ref_for_chunk(&chunk.chunk_id)
                    .unwrap()
                    .payload
                    .offset
            };
            let original = fs::read(&path).unwrap();
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&[original[offset as usize] ^ 1]).unwrap();
            file.flush().unwrap();
            TEXT_BUILDS.with(|count| count.set(0));
            TEXT_CHECKPOINT_BUILDS.with(|count| count.set(0));
            if prepared {
                let mut inner = db.lock().unwrap();
                let mut document = inner
                    .catalog
                    .document_by_external_id(
                        &inner
                            .catalog
                            .collection_by_name("first")
                            .unwrap()
                            .collection_id,
                        "doc",
                    )
                    .unwrap()
                    .clone();
                document.title = Some("updated".into());
                let mut records = crate::storage::PreparedRecords::create(
                    vec![RecordPayload::DocumentPut(document)],
                    path.parent().unwrap(),
                    1_048_576,
                )
                .unwrap();
                let pending = inner
                    .commit_prepared_batch(records.take_batch(1).unwrap())
                    .unwrap();
                inner
                    .finish_index_update(pending, &db.json_context())
                    .unwrap();
            } else {
                db.collection("first")
                    .unwrap()
                    .insert_document(input("new", "shared fresh"))
                    .unwrap();
            }
            assert_eq!(TEXT_BUILDS.with(Cell::get), 0);
            assert_eq!(TEXT_CHECKPOINT_BUILDS.with(Cell::get), 1);
            assert!(!db.index_stats().unwrap().rebuild_needed);
            assert_eq!(db.temporary_work.used(), 0);
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&[original[offset as usize]]).unwrap();
            file.flush().unwrap();
            assert_checkpoint(&db);
            drop((file, db));
            let reopened = Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    cache_budget_mb: Some(cache),
                    ..options()
                },
            )
            .unwrap();
            assert_checkpoint(&reopened);
            drop(reopened);
            fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn zero_cache_writer_deferred_union_discards_a_later_changed_spool() {
    for cache in [0, 1] {
        let path = query_preflight_fixture("zero-writer-deferred", "shared old");
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(cache),
                ..options()
            },
        )
        .unwrap();
        assert_eq!(db.lock().unwrap().lazy_text, cache == 0);
        let context = db.json_context();
        {
            let mut inner = db.lock().unwrap();
            let calls = inner.index_checkpoint_refreshes;
            let first_id = inner
                .catalog
                .collection_by_name("first")
                .unwrap()
                .collection_id
                .clone();
            let second_id = inner
                .catalog
                .collection_by_name("second")
                .unwrap()
                .collection_id
                .clone();
            let mut pending = None;
            for owner in [&first_id, &second_id] {
                let mut chunk = inner
                    .catalog
                    .chunk_by_external_id(owner, "chunk-1")
                    .unwrap()
                    .clone();
                chunk.text = "new needle".into();
                chunk.text_hash = chunk_text_hash(&chunk.text);
                let mut records = crate::storage::PreparedRecords::create(
                    vec![RecordPayload::ChunkPut(chunk)],
                    path.parent().unwrap(),
                    1_048_576,
                )
                .unwrap();
                let next = inner
                    .commit_prepared_batch_tracking(
                        records.take_batch(1).unwrap(),
                        pending.as_mut(),
                    )
                    .unwrap();
                if pending.is_none() {
                    pending = Some(next);
                }
                assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
            }
            let mut records = crate::storage::PreparedRecords::create(
                vec![RecordPayload::CollectionDrop {
                    collection_id: first_id.clone(),
                    name: "first".into(),
                    reason: Some("test drop".into()),
                    dropped_at: "1970-01-01T00:00:00Z".into(),
                }],
                path.parent().unwrap(),
                1_048_576,
            )
            .unwrap();
            drop(
                inner
                    .commit_prepared_batch_tracking(
                        records.take_batch(1).unwrap(),
                        pending.as_mut(),
                    )
                    .unwrap(),
            );
            inner
                .finish_index_update(pending.take().unwrap(), &context)
                .unwrap();
            assert_eq!(inner.index_checkpoint_refreshes, calls + 1);
            assert_eq!(inner.catalog.live_collections().count(), 2);
        }
        assert_checkpoint(&db);
        assert_eq!(
            db.collection("second")
                .unwrap()
                .search()
                .text("needle")
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        drop(db);
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn zero_cache_writer_prepare_failure_preserves_primary_and_compact_publication_boundaries() {
    for cache in [0, 1] {
        let path = query_preflight_fixture("zero-writer-failure", "shared needle");
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(cache),
                ..options()
            },
        )
        .unwrap();
        assert_eq!(db.lock().unwrap().lazy_text, cache == 0);
        {
            let mut inner = db.lock().unwrap();
            inner.temporary_work = Arc::new(crate::temporary::TemporaryDiskBudget::new(0));
        }
        let original = fs::read(&path).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        let error = db
            .collection("first")
            .unwrap()
            .insert_document(input("refused", "new"))
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(db.stats().unwrap().last_commit_id, commit);
        assert_eq!(
            db.compact(CompactOptions {
                rebuild_indexes: true
            })
            .unwrap_err()
            .code(),
            ErrorCode::ResourceExhausted
        );
        assert_eq!(fs::read(&path).unwrap(), original);
        db.compact(CompactOptions {
            rebuild_indexes: false,
        })
        .unwrap();
        assert_eq!(
            db.lock().unwrap().index_suffix_status,
            IndexSuffixState::Missing
        );
        db.lock().unwrap().temporary_work = db.temporary_work.clone();
        db.rebuild_index(None).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        let calls = db.lock().unwrap().index_checkpoint_refreshes;
        fail_next_index_checkpoint_refresh(&db);
        db.collection("first")
            .unwrap()
            .insert_document(input("committed", "new shared"))
            .unwrap();
        assert_eq!(db.lock().unwrap().index_checkpoint_refreshes, calls + 1);
        assert!(db.index_stats().unwrap().rebuild_needed);
        assert_eq!(db.stats().unwrap().last_commit_id, commit + 1);
        assert!(db
            .collection("first")
            .unwrap()
            .get_document_by_external_id("committed")
            .is_ok());
        assert_eq!(db.temporary_work.used(), 0);
        db.rebuild_index(None).unwrap();
        let calls = db.lock().unwrap().index_checkpoint_refreshes;
        let commit = db.stats().unwrap().last_commit_id;
        fail_next_index_checkpoint_refresh(&db);
        let summary = db
            .compact(CompactOptions {
                rebuild_indexes: true,
            })
            .unwrap();
        assert_eq!(db.lock().unwrap().index_checkpoint_refreshes, calls + 1);
        assert_eq!(summary.index_rebuild, "failed");
        assert_eq!(db.stats().unwrap().document_count, 3);
        let published_commit = db.stats().unwrap().last_commit_id;
        assert!(published_commit > commit);
        assert_eq!(db.temporary_work.used(), 0);
        let reopened = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                ..options()
            },
        )
        .unwrap();
        assert_eq!(reopened.stats().unwrap().last_commit_id, published_commit);
        assert_eq!(reopened.stats().unwrap().document_count, 3);
        assert!(reopened
            .collection("first")
            .unwrap()
            .get_document_by_external_id("committed")
            .is_ok());
        let error = reopened
            .collection("first")
            .unwrap()
            .search()
            .text("new")
            .execute()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IndexCorruption);
        assert_eq!(
            error_detail_value(&error, "index_unavailable_reason"),
            Some("transient_rebuild_budget_exceeded")
        );
        drop(reopened);
        let reference = Database::open_read_only(&path).unwrap();
        assert_eq!(
            reference
                .collection("first")
                .unwrap()
                .search()
                .text("new")
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        drop(reference);
        db.rebuild_index(None).unwrap();
        assert_checkpoint(&db);
        drop(db);
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn zero_cache_writer_rebuilds_after_old_suffix_truncation_without_stale_sources() {
    let path = query_preflight_fixture("zero-writer-old-eof", "shared needle");
    let db = Database::open_with_options(&path, options()).unwrap();
    let end = db.lock().unwrap().storage.critical_end_offset();
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(end)
        .unwrap();
    db.collection("first")
        .unwrap()
        .insert_document(input("new", "new shared"))
        .unwrap();
    assert_checkpoint(&db);
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn writer_healthy_reuse_keeps_df_zero_dictionary_carriers_through_compact() {
    for cache in [0, 1] {
        healthy_reuse_keeps_df_zero(cache);
    }
}

type PreparedTextEvents = BTreeMap<String, Vec<(u8, String, u64, usize)>>;

fn prepared_text_events(
    inner: &Inner,
    prepared: &TextCheckpoint,
    context: &crate::json::JsonOperationContext,
) -> PreparedTextEvents {
    use crate::storage::TextSegmentEntry;
    inner
        .catalog
        .live_collections()
        .map(|collection| {
            let mut events = Vec::new();
            prepared
                .body(&collection.collection_id)
                .unwrap()
                .visit(context, 0, &mut |entry| {
                    events.push(match entry {
                        TextSegmentEntry::Header { chunks } => (0, String::new(), 0, chunks),
                        TextSegmentEntry::Chunk { id, length } => (1, String::new(), id, length),
                        TextSegmentEntry::Term {
                            text,
                            document_frequency,
                        } => (2, text.to_owned(), 0, document_frequency),
                        TextSegmentEntry::Posting {
                            term,
                            chunk,
                            frequency,
                        } => (3, term.to_owned(), chunk, frequency),
                    });
                    Ok(())
                })
                .unwrap();
            (collection.collection_id.clone(), events)
        })
        .collect()
}

fn healthy_reuse_keeps_df_zero(cache: usize) {
    use crate::storage::{
        TextIndexChunkEntry, TextIndexPostingEntry, TextIndexSegmentPayload, TextIndexTermEntry,
    };
    let path = query_preflight_fixture("zero-writer-df-zero", "alpha");
    let db = Database::open(&path).unwrap();
    {
        let mut inner = db.lock().unwrap();
        let segments = inner
            .catalog
            .live_collections()
            .map(|collection| {
                let chunks = inner
                    .catalog
                    .live_chunks()
                    .filter(|chunk| chunk.collection_id == collection.collection_id)
                    .map(|chunk| TextIndexChunkEntry {
                        chunk_id: chunk.chunk_id.clone(),
                        document_id: chunk.document_id.clone(),
                        token_count: 1,
                    })
                    .collect::<Vec<_>>();
                let postings = chunks
                    .iter()
                    .map(|chunk| TextIndexPostingEntry {
                        chunk_id: chunk.chunk_id.clone(),
                        term_frequency: 1,
                    })
                    .collect::<Vec<_>>();
                let mut terms = Vec::new();
                if !chunks.is_empty() {
                    terms.push(TextIndexTermEntry {
                        term: "alpha".into(),
                        postings: postings.clone(),
                    });
                }
                if collection.name == "first" {
                    terms.push(TextIndexTermEntry {
                        term: "empty".into(),
                        postings: Vec::new(),
                    });
                }
                if collection.name == "second" {
                    terms.push(TextIndexTermEntry {
                        term: "beta".into(),
                        postings,
                    });
                }
                TextIndexSegmentPayload {
                    collection_id: collection.collection_id.clone(),
                    covered_commit_sequence_no: inner.last_commit_sequence_no,
                    chunks,
                    terms,
                }
            })
            .collect();
        let checkpoint = IndexCheckpointPayload {
            metadata: inner.index_checkpoint_metadata(&db.json_context()).unwrap(),
            text_segments: segments,
            vector_segments: inner
                .vector_index_segments(BTreeMap::new(), &BTreeSet::new())
                .unwrap(),
        };
        inner
            .storage
            .write_index_checkpoint(&checkpoint, Durability::Fast)
            .unwrap();
    }
    drop(db);
    let check = |db: &Database, retained| {
        assert_eq!(db.index_stats().unwrap().text_terms, 3);
        assert_eq!(
            db.collection("first")
                .unwrap()
                .stats()
                .unwrap()
                .text_index_terms,
            1
        );
        assert_eq!(
            db.collection("second")
                .unwrap()
                .stats()
                .unwrap()
                .text_index_terms,
            2
        );
        assert_eq!(
            db.collection("empty")
                .unwrap()
                .stats()
                .unwrap()
                .text_index_terms,
            0
        );
        assert_eq!(
            db.lock().unwrap().text_index.has_resident_cache(),
            cache != 0
        );
        assert_eq!(db.temporary_work.used(), retained);
    };
    let open_options = DatabaseOpenOptions {
        cache_budget_mb: Some(cache),
        ..options()
    };
    let db = Database::open_with_options(&path, open_options.clone()).unwrap();
    check(&db, 0);
    db.put_embeddings_in_collection(
        db.collection("second").unwrap().collection_id(),
        vec![EmbeddingInput {
            chunk_external_id: "chunk-1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![0.0, 1.0],
        }],
        OnConflict::Replace,
    )
    .unwrap();
    check(&db, 0);
    // Only this empty collection changes; both dictionary carriers are unchanged.
    let empty = db.collection("empty").unwrap();
    let mut inner = db.lock().unwrap();
    let mut document = inner
        .catalog
        .document_by_external_id(
            &inner
                .catalog
                .collection_by_name("first")
                .unwrap()
                .collection_id,
            "doc",
        )
        .unwrap()
        .clone();
    document.document_id = inner.catalog.next_id("doc_", 0).unwrap();
    document.collection_id = empty.collection_id().into();
    document.external_id = "metadata-only".into();
    let pending = inner
        .commit_records(vec![RecordPayload::DocumentPut(document)])
        .unwrap();
    inner
        .finish_index_update(pending, &db.json_context())
        .unwrap();
    drop(inner);
    check(&db, 0);
    let context = db.json_context();
    let before = {
        let inner = db.lock().unwrap();
        crate::tokenizer::TOKEN_VISITS.with(|count| count.set(0));
        let prepared = inner
            .prepare_compact_text_checkpoint(&context)
            .unwrap()
            .unwrap();
        assert!(prepared.has_zero_frequency_terms());
        let events = prepared_text_events(&inner, &prepared, &context);
        assert_eq!(crate::tokenizer::TOKEN_VISITS.with(Cell::get), 0);
        events
    };
    assert_eq!(db.temporary_work.used(), 0);
    let query = |db: &Database| {
        db.collection("second")
            .unwrap()
            .search()
            .text("alpha beta")
            .execute()
            .unwrap()
            .results
            .into_iter()
            .map(|hit| {
                (
                    hit.chunk_id,
                    hit.final_score.to_bits(),
                    hit.bm25_score.map(f64::to_bits),
                )
            })
            .collect::<Vec<_>>()
    };
    let expected = query(&db);
    let reader = Database::open_read_only(&path).unwrap();
    assert_eq!(query(&reader), expected);
    let old_commit = db.stats().unwrap().last_commit_id;
    assert_eq!(reader.stats().unwrap().last_commit_id, old_commit);
    let retained = fs::metadata(&path).unwrap().len();
    let summary = db
        .compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
    assert_eq!(summary.index_rebuild, "full_checkpoint_written");
    assert!(db.stats().unwrap().last_commit_id > old_commit);
    check(&db, retained);
    assert_eq!(query(&db), expected);
    assert_eq!(query(&reader), expected);
    assert_eq!(reader.stats().unwrap().last_commit_id, old_commit);
    {
        let mut inner = db.lock().unwrap();
        let prepared = inner
            .prepare_compact_text_checkpoint(&context)
            .unwrap()
            .unwrap();
        assert_eq!(prepared_text_events(&inner, &prepared, &context), before);
        drop(prepared);
        // Counts are unchanged, but a stale covered commit cannot authorize reuse.
        let commit = inner.last_commit_sequence_no;
        inner.last_commit_sequence_no += 1;
        crate::tokenizer::TOKEN_VISITS.with(|count| count.set(0));
        let mut rebuilt = inner
            .prepare_compact_text_checkpoint(&context)
            .unwrap()
            .unwrap();
        assert_eq!(rebuilt.take_stats().term_count(), 1);
        assert!(crate::tokenizer::TOKEN_VISITS.with(Cell::get) > 0);
        drop(rebuilt);
        inner.last_commit_sequence_no = commit;
    }
    let original = fs::read(&path).unwrap();
    let beta = original
        .windows(4)
        .position(|bytes| bytes == b"beta")
        .unwrap();
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(beta as u64)).unwrap();
    file.write_all(b"z").unwrap();
    {
        let inner = db.lock().unwrap();
        let mut repaired = inner
            .prepare_compact_text_checkpoint(&context)
            .unwrap()
            .unwrap();
        let events = prepared_text_events(&inner, &repaired, &context);
        let first = inner.catalog.collection_by_name("first").unwrap();
        let second = inner.catalog.collection_by_name("second").unwrap();
        assert_eq!(events[&first.collection_id], before[&first.collection_id]);
        assert_ne!(events[&second.collection_id], before[&second.collection_id]);
        assert_eq!(repaired.take_stats().term_count(), 2);
    }
    file.seek(SeekFrom::Start(beta as u64)).unwrap();
    file.write_all(b"b").unwrap();
    drop(file);
    assert_eq!(fs::read(&path).unwrap(), original);
    check(&db, retained);
    drop(reader);
    assert_eq!(db.temporary_work.used(), 0);
    drop((empty, db));
    let reopened = Database::open_with_options(&path, open_options).unwrap();
    check(&reopened, 0);
    assert!(scan_file(&path).unwrap().findings.is_empty());
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn zero_cache_writer_suffix_attempts_keep_spools_and_summary_through_exact_budget_retry() {
    let source = query_preflight_fixture("zero-writer-suffix-budget-source", "shared needle");
    for parent in [false, true] {
        for delta in [-1isize, 0, 1] {
            let path = temp_path("zero-writer-suffix-budget");
            fs::copy(&source, &path).unwrap();
            let db = Database::open_with_options(&path, options()).unwrap();
            let context = db.json_context();
            let mut inner = db.lock().unwrap();
            let old_source_bytes = inner.lazy_vector_source.as_ref().unwrap().reserved_bytes();
            let original_available = context.available_workspace_bytes();
            let primary_end = inner.storage.critical_end_offset() as usize;
            let primary = fs::read(&path).unwrap()[..primary_end].to_vec();
            let signature = inner.catalog.state_signature();
            let commit = inner.catalog.last_commit_id;
            let mut text = inner
                .prepare_text_checkpoint(None, &BTreeSet::new(), &context)
                .unwrap()
                .unwrap();
            let identities = inner
                .catalog
                .live_collections()
                .map(|collection| {
                    (
                        collection.collection_id.clone(),
                        text.body(&collection.collection_id).unwrap().extent as *const _ as usize,
                    )
                })
                .collect::<Vec<_>>();
            let disk = inner.temporary_work.used();
            assert!(disk > 0);
            inner.mark_index_checkpoint_rebuild_needed();
            let count = inner.catalog.live_collections().count();
            let directory = (2 * count + 1) * (5 * 8 + 4 + 2usize).next_multiple_of(8);
            let coverage = count
                * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
            let peak = inner
                .catalog
                .live_collections()
                .map(|collection| {
                    let embeddings = inner
                        .catalog
                        .live_embeddings()
                        .filter(|entry| entry.collection_id == collection.collection_id);
                    let array = embeddings.clone().count()
                        * std::mem::size_of::<(&Embedding, Option<&crate::storage::RecordRef>)>();
                    let decoded = embeddings
                        .map(|entry| {
                            entry.dim * 4
                                + entry.embedding_id.len()
                                + entry.chunk_id.len()
                                + entry.collection_id.len()
                                + entry.model_name.len()
                                + entry.created_at.len()
                        })
                        .max()
                        .unwrap_or(0);
                    array + decoded
                })
                .max()
                .unwrap();
            let minimum = coverage.max(directory + peak);
            let available = (minimum as isize + delta) as usize;
            let (held, parent_held) = pressure(&inner, &context, available, parent);
            let calls = inner.index_checkpoint_refreshes;
            let first = inner
                .refresh_index_checkpoint_attempt(
                    VectorCache::new(&context),
                    &BTreeSet::new(),
                    true,
                    Some(&mut text),
                    &context,
                )
                .unwrap_err();
            assert!(crate::json::is_memory_budget_error(&first));
            assert_eq!(inner.index_checkpoint_refreshes, calls + 1);
            assert_eq!(inner.temporary_work.used(), disk);
            assert_eq!(fs::read(&path).unwrap()[..primary_end], primary);
            let result = inner.refresh_index_checkpoint_attempt(
                VectorCache::new(&context),
                &BTreeSet::new(),
                false,
                Some(&mut text),
                &context,
            );
            if delta < 0 {
                assert!(crate::json::is_memory_budget_error(&result.unwrap_err()));
                assert!(inner.lazy_vector_source.is_none());
            } else {
                assert_eq!(result.unwrap().unwrap().status, "persistent_checkpoint");
            }
            assert_eq!(inner.index_checkpoint_refreshes, calls + 2);
            for (id, identity) in identities {
                assert_eq!(
                    text.body(&id).unwrap().extent as *const _ as usize,
                    identity
                );
            }
            assert_eq!(inner.temporary_work.used(), disk);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.catalog.last_commit_id, commit);
            assert_eq!(fs::read(&path).unwrap()[..primary_end], primary);
            drop(text);
            assert_eq!(inner.temporary_work.used(), 0);
            drop((held, parent_held));
            let new_source_bytes = inner
                .lazy_vector_source
                .as_ref()
                .map_or(0, IndexSegmentSource::reserved_bytes);
            assert_eq!(
                context.available_workspace_bytes(),
                original_available + old_source_bytes - new_source_bytes
            );
            // Exercise the real production wrapper, not only the attempt helper.
            let text = inner
                .prepare_text_checkpoint(None, &BTreeSet::new(), &context)
                .unwrap();
            inner.mark_index_checkpoint_rebuild_needed();
            let (held, parent_held) = pressure(&inner, &context, available, parent);
            let calls = inner.index_checkpoint_refreshes;
            let result = inner.refresh_index_checkpoint_prepared(
                VectorCache::new(&context),
                &BTreeSet::new(),
                text,
                &context,
            );
            assert_eq!(inner.index_checkpoint_refreshes, calls + 2);
            if delta < 0 {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                let limit = if parent {
                    3 * 1024 * 1024 * 1024usize
                } else {
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                };
                for (key, value) in [
                    ("resource", "memory".to_owned()),
                    ("operation", "operation_scratch".to_owned()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", (limit + 1).to_string()),
                ] {
                    assert_eq!(error_detail_value(&error, key), Some(value.as_str()));
                }
                assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
            } else {
                assert_eq!(result.unwrap().unwrap().status, "persistent_checkpoint");
                assert_eq!(inner.text_stats().unwrap().term_count(), 2);
            }
            assert_eq!(inner.temporary_work.used(), 0);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.catalog.last_commit_id, commit);
            assert_eq!(fs::read(&path).unwrap()[..primary_end], primary);
            drop((held, parent_held));
            let new_source_bytes = inner
                .lazy_vector_source
                .as_ref()
                .map_or(0, IndexSegmentSource::reserved_bytes);
            assert_eq!(
                context.available_workspace_bytes(),
                original_available + old_source_bytes - new_source_bytes
            );
            drop(inner);
            if delta < 0 {
                db.rebuild_index(None).unwrap();
            }
            assert_checkpoint(&db);
            drop(db);
            fs::remove_file(path).unwrap();
        }
    }
    fs::remove_file(source).unwrap();
}

#[test]
fn compact_text_reuse_budget_fallback_drops_partial_staging_and_preserves_native_errors() {
    use crate::storage::{TextIndexSegmentPayload, TextIndexTermEntry};
    let directory = temp_path("compact-text-reuse-budget");
    fs::create_dir(&directory).unwrap();
    let path = directory.join("data.coffer");
    let db = Database::create(
        &path,
        DatabaseConfig {
            cache_budget_mb: Some(0),
            durability: Durability::Fast,
            ..Default::default()
        },
    )
    .unwrap();
    for name in ["first", "second"] {
        db.create_collection(CollectionConfig {
            name: name.into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    }
    const TERM_BYTES: usize = 4097;
    {
        let mut inner = db.lock().unwrap();
        let checkpoint = IndexCheckpointPayload {
            metadata: inner.index_checkpoint_metadata(&db.json_context()).unwrap(),
            text_segments: inner
                .catalog
                .live_collections()
                .map(|collection| TextIndexSegmentPayload {
                    collection_id: collection.collection_id.clone(),
                    covered_commit_sequence_no: inner.last_commit_sequence_no,
                    chunks: Vec::new(),
                    terms: if collection.name == "second" {
                        vec![TextIndexTermEntry {
                            term: "x".repeat(TERM_BYTES),
                            postings: Vec::new(),
                        }]
                    } else {
                        Vec::new()
                    },
                })
                .collect(),
            vector_segments: inner
                .vector_index_segments(BTreeMap::new(), &BTreeSet::new())
                .unwrap(),
        };
        inner
            .storage
            .write_index_checkpoint(&checkpoint, Durability::Fast)
            .unwrap();
    }
    drop(db);
    let db = Database::open_with_options(&path, options()).unwrap();
    let context = db.json_context();
    let available = context.available_workspace_bytes();
    let original = fs::read(&path).unwrap();
    let entries = fs::read_dir(&directory).unwrap().count();
    let mut inner = db.lock().unwrap();
    assert!(inner.text_stats().is_some());
    assert_eq!(inner.text_stats().unwrap().term_count(), 1);
    let management =
        crate::temporary::test_file_management_bytes(inner.storage.path().parent().unwrap());
    let needed = management + 2 * std::mem::size_of::<crate::storage::TextExtent>() + TERM_BYTES;
    for parent in [false, true] {
        for allowance in [needed - 1, needed, needed + 1] {
            let (held, parent_held) = pressure(&inner, &context, allowance, parent);
            let result = inner.prefetch_unchanged_text(&BTreeSet::new(), &context);
            if allowance < needed {
                let error = result.err().unwrap();
                let limit = if parent {
                    3 * 1024 * 1024 * 1024usize
                } else {
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                };
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
                assert_eq!(
                    error_detail_value(&error, "limit_bytes"),
                    Some(limit.to_string().as_str())
                );
                assert_eq!(
                    error_detail_value(&error, "requested_bytes"),
                    Some((limit + 1).to_string().as_str())
                );
            } else {
                drop(result.unwrap().unwrap());
            }
            assert_eq!(context.available_workspace_bytes(), allowance);
            assert_eq!(db.temporary_work.used(), 0);
            assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
            TEXT_CHECKPOINT_BUILDS.with(|count| count.set(0));
            let mut prepared = inner
                .prepare_compact_text_checkpoint(&context)
                .unwrap()
                .unwrap();
            // At M/M+1 capture fits, but finishing the reused checkpoint needs extra credit.
            assert_eq!(
                TEXT_CHECKPOINT_BUILDS.with(Cell::get),
                if allowance < needed { 1 } else { 2 }
            );
            assert_eq!(prepared.take_stats().term_count(), 0);
            drop(prepared);
            assert_eq!(context.available_workspace_bytes(), allowance);
            assert_eq!(db.temporary_work.used(), 0);
            drop((held, parent_held));
            assert_eq!(context.available_workspace_bytes(), available);
        }
    }
    TEXT_CHECKPOINT_BUILDS.with(|count| count.set(0));
    let error = crate::test_alloc::fail_byte_allocation_once(TERM_BYTES, || {
        inner.prepare_compact_text_checkpoint(&context)
    })
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(TEXT_CHECKPOINT_BUILDS.with(Cell::get), 0);
    assert_eq!(context.available_workspace_bytes(), available);
    assert_eq!(db.temporary_work.used(), 0);
    let previous_disk = inner.temporary_work.clone();
    let disk = Arc::new(crate::temporary::TemporaryDiskBudget::new(100));
    inner.temporary_work = disk.clone();
    let mut prepared = inner
        .prepare_compact_text_checkpoint(&context)
        .unwrap()
        .unwrap();
    assert_eq!(prepared.take_stats().term_count(), 0);
    assert_eq!(disk.used(), 100);
    drop(prepared);
    assert_eq!(disk.used(), 0);
    inner.temporary_work = previous_disk;
    assert_eq!(context.available_workspace_bytes(), available);
    assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
    assert_eq!(fs::read(&path).unwrap(), original);
    drop(inner);
    drop(db);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn resident_incremental_update_restores_a_term_that_becomes_only_a_df_zero_carrier() {
    use crate::storage::{
        TextIndexChunkEntry, TextIndexPostingEntry, TextIndexSegmentPayload, TextIndexTermEntry,
    };
    let path = query_preflight_fixture("resident-df-zero-after-delete", "alpha");
    let db = Database::open(&path).unwrap();
    {
        let mut inner = db.lock().unwrap();
        let text_segments = inner
            .catalog
            .live_collections()
            .map(|collection| {
                let chunks = inner
                    .catalog
                    .live_chunks()
                    .filter(|chunk| chunk.collection_id == collection.collection_id)
                    .map(|chunk| TextIndexChunkEntry {
                        chunk_id: chunk.chunk_id.clone(),
                        document_id: chunk.document_id.clone(),
                        token_count: 1,
                    })
                    .collect::<Vec<_>>();
                let postings = chunks
                    .iter()
                    .map(|chunk| TextIndexPostingEntry {
                        chunk_id: chunk.chunk_id.clone(),
                        term_frequency: 1,
                    })
                    .collect::<Vec<_>>();
                let terms = match collection.name.as_str() {
                    "first" => vec![TextIndexTermEntry {
                        term: "alpha".into(),
                        postings,
                    }],
                    "second" => vec![
                        TextIndexTermEntry {
                            term: "alpha".into(),
                            postings: Vec::new(),
                        },
                        TextIndexTermEntry {
                            term: "beta".into(),
                            postings,
                        },
                    ],
                    _ => Vec::new(),
                };
                TextIndexSegmentPayload {
                    collection_id: collection.collection_id.clone(),
                    covered_commit_sequence_no: inner.last_commit_sequence_no,
                    chunks,
                    terms,
                }
            })
            .collect();
        let checkpoint = IndexCheckpointPayload {
            metadata: inner.index_checkpoint_metadata(&db.json_context()).unwrap(),
            text_segments,
            vector_segments: inner
                .vector_index_segments(BTreeMap::new(), &BTreeSet::new())
                .unwrap(),
        };
        inner
            .storage
            .write_index_checkpoint(&checkpoint, Durability::Fast)
            .unwrap();
    }
    drop(db);
    let db = Database::open(&path).unwrap();
    let (beta_pointer, beta_owner) = {
        let inner = db.lock().unwrap();
        assert!(inner.text_index.has_resident_cache());
        assert!(inner
            .text_index
            .postings()
            .values()
            .all(|postings| !postings.is_empty()));
        let beta = &inner.text_index.postings()["beta"];
        (beta.as_ptr(), Arc::as_ptr(&beta[0].collection_id))
    };
    let first = db.collection("first").unwrap();
    let document = first.get_document_by_external_id("doc").unwrap();
    first.delete_document(&document.document_id).unwrap();
    {
        let inner = db.lock().unwrap();
        assert!(inner.text_index.has_resident_cache());
        assert!(inner.text_index.postings()["alpha"].is_empty());
        let beta = &inner.text_index.postings()["beta"];
        assert_eq!(beta.as_ptr(), beta_pointer);
        assert_eq!(Arc::as_ptr(&beta[0].collection_id), beta_owner);
    }
    assert_eq!(db.index_stats().unwrap().text_terms, 2);
    assert_eq!(
        db.collection("second")
            .unwrap()
            .stats()
            .unwrap()
            .text_index_terms,
        1
    );
    assert!(db
        .collection("second")
        .unwrap()
        .search()
        .text("alpha")
        .execute()
        .unwrap()
        .results
        .is_empty());
    assert_eq!(
        db.collection("second")
            .unwrap()
            .search()
            .text("beta")
            .execute()
            .unwrap()
            .results
            .len(),
        4
    );
    let second = db.collection("second").unwrap();
    let document = second.get_document_by_external_id("doc").unwrap();
    second.delete_document(&document.document_id).unwrap();
    assert_eq!(db.index_stats().unwrap().text_terms, 0);
    assert!(db.lock().unwrap().text_index.postings().is_empty());
    drop((first, second, db));
    let db = Database::open(&path).unwrap();
    assert_eq!(db.index_stats().unwrap().text_terms, 0);
    assert!(db.lock().unwrap().text_index.has_resident_cache());
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn nonzero_text_cache_miss_keeps_healthy_source_and_public_query_results() {
    let text = (0..3000).map(|n| format!("token{n} ")).collect::<String>();
    let path = query_preflight_fixture("nonzero-text-cache-miss", &text);
    let before = fs::read(&path).unwrap();
    let db = Database::open_with_options(
        &path,
        DatabaseOpenOptions {
            cache_budget_mb: Some(1),
            ..options()
        },
    )
    .unwrap();
    {
        let inner = db.lock().unwrap();
        assert!(!inner.lazy_text);
        assert!(!inner.text_index.has_resident_cache());
        assert_eq!(inner.text_index.reserved_bytes(), 0);
        assert!(inner.lazy_vector_source.is_some());
        assert_eq!(
            inner.index_suffix_status,
            IndexSuffixState::PersistentCheckpoint
        );
    }
    assert_eq!(db.index_stats().unwrap().text_terms, 3000);
    let first = db.collection("first").unwrap();
    let result = first.search().text("token7").execute().unwrap();
    assert_eq!(result.results.len(), 2);
    assert_eq!(result.query_context.index_source, "persistent_checkpoint");
    assert!(!result.query_context.transient_rebuild.started);
    assert!(result
        .results
        .iter()
        .all(|result| result.explain.get("scores").is_some()));
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(db.temporary_work.used(), 0);
    drop((result, first, db));
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(unix)]
fn zero_cache_writer_prefetch_admits_file_owner_before_optional_buffers() {
    let path = query_preflight_fixture("zero-writer-prefetch-owner-budget", "shared needle");
    let db = Database::open_with_options(&path, options()).unwrap();
    let context = db.json_context();
    let inner = db.lock().unwrap();
    let source = inner.lazy_vector_source.as_ref().unwrap();
    let empty = inner.catalog.collection_by_name("empty").unwrap();
    let original = fs::read(&path).unwrap();
    let available = context.available_workspace_bytes();
    let management =
        crate::temporary::test_file_management_bytes(inner.storage.path().parent().unwrap());
    // The filename is temporary, but coexists with both retained path owners.
    let needed = management + 128;
    for parent in [false, true] {
        for allowance in [needed - 1, needed, needed + 1] {
            let (held, parent_held) = pressure(&inner, &context, allowance, parent);
            let result = source.stage_text(
                &inner.storage,
                &empty.collection_id,
                &db.temporary_work,
                &context,
            );
            if allowance < needed {
                let error = result.err().unwrap();
                let limit = if parent {
                    3 * 1024 * 1024 * 1024usize
                } else {
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                };
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "memory".to_owned()),
                    ("operation", "operation_scratch".to_owned()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", (limit + 1).to_string()),
                ] {
                    assert_eq!(error_detail_value(&error, key), Some(value.as_str()));
                }
            } else {
                let body = result.unwrap().unwrap();
                assert_eq!(context.available_workspace_bytes(), allowance - management);
                assert_eq!(db.temporary_work.used(), 50);
                drop(body);
            }
            assert_eq!(context.available_workspace_bytes(), allowance);
            assert_eq!(db.temporary_work.used(), 0);
            drop((held, parent_held));
            assert_eq!(context.available_workspace_bytes(), available);
        }
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    drop(inner);
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn zero_cache_writer_postcommit_necessary_prepare_error_is_not_swallowed() {
    for cache in [0, 1] {
        let path = query_preflight_fixture("zero-writer-postcommit-prepare", "shared needle");
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(cache),
                ..options()
            },
        )
        .unwrap();
        assert_eq!(db.lock().unwrap().lazy_text, cache == 0);
        let context = db.json_context();
        let mut inner = db.lock().unwrap();
        let first = inner.catalog.collection_by_name("first").unwrap();
        let mut document = inner
            .catalog
            .document_by_external_id(&first.collection_id, "doc")
            .unwrap()
            .clone();
        document.title = Some("committed title".into());
        let batch =
            OwnedRecordBatch::single(RecordPayload::DocumentPut(document), &context).unwrap();
        let pending = inner.commit_records_in(batch).unwrap();
        let committed = fs::read(&path).unwrap();
        let commit = inner.catalog.last_commit_id;
        let calls = inner.index_checkpoint_refreshes;
        let limit = db.temporary_work.limit_bytes();
        let occupied = limit - db.temporary_work.used();
        let (blocker_file, mut blocker) =
            crate::temporary::create_budgeted_temporary_file_in_context(
                inner.storage.path().parent().unwrap(),
                "postcommit-budget",
                "tmp",
                db.temporary_work.clone(),
                &context,
                "operation_scratch",
            )
            .unwrap();
        blocker.reserve_disk(occupied).unwrap();
        let error = inner.finish_index_update(pending, &context).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        for (key, expected) in [
            ("resource", "temporary_disk".to_owned()),
            ("operation", "temporary_staging".to_owned()),
            ("limit_bytes", limit.to_string()),
            ("requested_bytes", (u128::from(limit) + 2).to_string()),
        ] {
            assert_eq!(error_detail_value(&error, key), Some(expected.as_str()));
        }
        assert_eq!(inner.index_checkpoint_refreshes, calls);
        assert_eq!(inner.catalog.last_commit_id, commit);
        assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
        assert!(inner.lazy_vector_source.is_none());
        assert_eq!(fs::read(&path).unwrap(), committed);
        assert_eq!(db.temporary_work.used(), occupied);
        drop((blocker_file, blocker));
        assert_eq!(db.temporary_work.used(), 0);
        drop(inner);
        assert_eq!(
            db.collection("first")
                .unwrap()
                .get_document_by_external_id("doc")
                .unwrap()
                .title
                .as_deref(),
            Some("committed title")
        );
        let reopened = Database::open_read_only(&path).unwrap();
        assert_eq!(reopened.stats().unwrap().last_commit_id, commit);
        assert_eq!(
            reopened
                .collection("first")
                .unwrap()
                .get_document_by_external_id("doc")
                .unwrap()
                .title
                .as_deref(),
            Some("committed title")
        );
        drop(reopened);
        db.rebuild_index(None).unwrap();
        assert_checkpoint(&db);
        drop(db);
        fs::remove_file(path).unwrap();
    }
}

#[test]
#[cfg(unix)]
fn zero_cache_writer_staging_descriptors_are_collection_independent_in_isolated_process() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "db::tests::zero_text_writer::zero_cache_writer_descriptor_worker",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("COFFERDB_TEXT_DESCRIPTOR_WORKER", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("descriptor-case:3:verified"));
    assert!(stdout.contains("descriptor-case:512:verified"));
}

#[test]
#[cfg(unix)]
#[ignore = "invoked by the descriptor isolation test in its own process"]
fn zero_cache_writer_descriptor_worker() {
    assert_eq!(
        std::env::var("COFFERDB_TEXT_DESCRIPTOR_WORKER").unwrap(),
        "1"
    );
    let descriptors = || {
        fs::read_dir("/dev/fd")
            .unwrap()
            .inspect(|entry| {
                entry.as_ref().unwrap();
            })
            .count()
    };
    for count in [3, 512] {
        let path = temp_path("zero-writer-descriptor-isolation");
        let db = Database::create(
            &path,
            DatabaseConfig {
                cache_budget_mb: Some(0),
                durability: Durability::Fast,
                ..Default::default()
            },
        )
        .unwrap();
        let context = db.json_context();
        let mut inner = db.lock().unwrap();
        let records = (1..=count)
            .map(|id| {
                RecordPayload::CollectionCreate(CollectionInfo {
                    collection_id: format!("col_{id:016}"),
                    name: format!("empty-{id}"),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: Some("m".into()),
                    tokenizer: Default::default(),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    created_at: "1970-01-01T00:00:00Z".into(),
                })
            })
            .collect();
        let pending = inner.commit_records(records).unwrap();
        inner.finish_index_update(pending, &context).unwrap();
        let baseline = descriptors();
        let reusable = inner
            .prefetch_unchanged_text(&BTreeSet::new(), &context)
            .unwrap()
            .unwrap();
        assert_eq!(descriptors(), baseline + 1);
        assert_eq!(db.temporary_work.used(), 50 * count as u64);
        let text = inner
            .prepare_text_checkpoint(Some(reusable), &BTreeSet::new(), &context)
            .unwrap()
            .unwrap();
        assert_eq!(descriptors(), baseline + 1);
        let first_id = "col_0000000000000001";
        let identity = text.body(first_id).unwrap().staging_identity().unwrap();
        for collection in inner.catalog.live_collections() {
            assert_eq!(
                text.body(&collection.collection_id)
                    .unwrap()
                    .staging_identity()
                    .unwrap(),
                identity
            );
        }
        inner.mark_index_checkpoint_rebuild_needed();
        let minimum = (2 * count + 1) * (5 * 8 + 4 + 2usize).next_multiple_of(8);
        let (held, parent_held) = pressure(&inner, &context, minimum - 1, false);
        let mut text = text;
        let error = inner
            .refresh_index_checkpoint_attempt(
                VectorCache::new(&context),
                &BTreeSet::new(),
                true,
                Some(&mut text),
                &context,
            )
            .unwrap_err();
        assert!(crate::json::is_memory_budget_error(&error));
        assert_eq!(descriptors(), baseline + 1);
        assert_eq!(
            text.body(first_id).unwrap().staging_identity().unwrap(),
            identity
        );
        drop((held, parent_held));
        let (held, parent_held) = pressure(&inner, &context, minimum, false);
        inner
            .refresh_index_checkpoint_attempt(
                VectorCache::new(&context),
                &BTreeSet::new(),
                false,
                Some(&mut text),
                &context,
            )
            .unwrap();
        assert_eq!(descriptors(), baseline + 1);
        assert_eq!(
            text.body(first_id).unwrap().staging_identity().unwrap(),
            identity
        );
        assert_eq!(inner.text_stats().unwrap().term_count(), 0);
        drop((held, parent_held, text));
        assert_eq!(descriptors(), baseline);
        assert_eq!(db.temporary_work.used(), 0);
        drop(inner);
        assert_eq!(
            Database::check_path(
                &path,
                CheckOptions {
                    verify_indexes: true
                }
            )
            .unwrap()
            .status,
            "ok"
        );
        drop(db);
        fs::remove_file(path).unwrap();
        println!("descriptor-case:{count}:verified");
    }
}
