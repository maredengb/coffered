use super::query_preparation::cache_for_preparation;
use super::*;
use crate::index::text::Bm25Hit;
use crate::index::vector::VectorHit;
use crate::json::{JsonOperationContext, JsonProcessingPolicyV1, ReservedVec};
use std::mem::size_of;

const TOTAL: usize = 3 * 1024 * 1024 * 1024;

fn assert_memory_error(error: &Error, limit: usize) {
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    assert_eq!(error_detail_value(error, "resource"), Some("memory"));
    assert_eq!(
        error_detail_value(error, "operation"),
        Some("operation_scratch")
    );
    assert_eq!(
        error_detail_value(error, "limit_bytes"),
        Some(limit.to_string().as_str())
    );
    assert_eq!(
        error_detail_value(error, "requested_bytes"),
        Some((limit + 1).to_string().as_str())
    );
}

fn owned_hits(context: &JsonOperationContext) -> (Bm25Hits, VectorHits, usize) {
    let baseline = context.available_workspace_bytes();
    let mut text = Bm25Hits::with_capacity(2, context).unwrap();
    text.push("a", 2.0, std::iter::empty()).unwrap();
    text.push("b", 1.0, std::iter::empty()).unwrap();
    let mut vector = VectorCandidates::new(2, context);
    vector.push("a", 0.5, 0.75).unwrap();
    vector.push("b", 0.0, 0.25).unwrap();
    let vector = vector.finish();
    let bytes = 2 * size_of::<Bm25Hit>() + 2 * size_of::<VectorHit>() + 4;
    assert_eq!(context.available_workspace_bytes(), baseline - bytes);
    (text, vector, bytes)
}

fn assert_fused(scores: &[CandidateScore<'_>], text: &[Bm25Hit]) {
    assert_eq!(scores.len(), 2);
    for (index, expected) in [0.875f64, 0.375].into_iter().enumerate() {
        assert_eq!(scores[index].chunk_id, text[index].chunk_id);
        assert_eq!(
            scores[index].chunk_id.as_ptr(),
            text[index].chunk_id.as_ptr()
        );
        assert_eq!(scores[index].final_score.to_bits(), expected.to_bits());
    }
}

#[test]
fn fusion_reclamation_preserves_owned_hits_at_both_exact_budget_boundaries() {
    let path = cached_vector_batch_fixture("fusion-reclaim-boundaries");
    let original = fs::read(&path).unwrap();
    let indices = 4 * size_of::<usize>();
    let ranked = 2 * size_of::<CandidateScore<'_>>();
    let peak = indices + ranked;
    for (parent, allowed) in [false, true]
        .into_iter()
        .flat_map(|parent| [indices - 1, peak - 1, peak, peak + 1].map(|allowed| (parent, allowed)))
    {
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let (_, directory, cached) = cache_for_preparation(&db);
        let (text, vector, hits_bytes) = owned_hits(&context);
        let pointers = (
            text.as_ptr(),
            vector.as_ptr(),
            text[0].chunk_id.as_ptr(),
            vector[0].chunk_id.as_ptr(),
        );
        let other = db.json_context();
        let mut retained = other.workspace_reservation();
        retained.grow_to(37, 37, "operation_scratch").unwrap();
        let blocker = parent.then(|| {
            context
                .cache_reservation(
                    (context.available_total_bytes() - allowed) as u128,
                    usize::MAX,
                )
                .unwrap()
        });
        let mut busy = other.workspace_reservation();
        if !parent {
            let occupied =
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory - hits_bytes - 37 - allowed;
            busy.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
        }
        assert_eq!(context.available_workspace_bytes(), allowed);
        let mut attempts = 0;
        let result = db.lock().unwrap().read_reclaiming(|inner| {
            attempts += 1;
            let result = fuse_candidates(
                &inner.catalog,
                &text,
                &vector,
                None,
                SearchWeights::new(0.5, 0.5, 0.0).unwrap(),
                2,
                &context,
            );
            if attempts == 1 && allowed < peak {
                assert_memory_error(
                    result.as_ref().err().unwrap(),
                    if parent {
                        TOTAL
                    } else {
                        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    },
                );
            }
            result
        });
        assert_eq!(attempts, if allowed < peak { 2 } else { 1 });
        if !parent && allowed < peak {
            let error = result.err().unwrap();
            assert_memory_error(&error, JsonProcessingPolicyV1::WORK_BUDGET_BYTES);
        } else {
            let scores = result.unwrap();
            assert_fused(scores.as_slice(), &text);
            assert_eq!(scores.reserved_bytes(), ranked);
            assert_eq!(
                context.available_workspace_bytes(),
                allowed + if parent && allowed < peak { cached } else { 0 } - ranked
            );
            drop(scores);
        }
        let reclaimed = if allowed < peak { cached } else { 0 };
        assert_eq!(
            context.available_workspace_bytes(),
            allowed + if parent { reclaimed } else { 0 }
        );
        assert_eq!(retained.reserved_bytes(), 37);
        assert_eq!(
            pointers,
            (
                text.as_ptr(),
                vector.as_ptr(),
                text[0].chunk_id.as_ptr(),
                vector[0].chunk_id.as_ptr()
            )
        );
        assert_eq!(text[0].score.to_bits(), 2.0f64.to_bits());
        assert_eq!(vector[0].normalized_score.to_bits(), 0.75f64.to_bits());
        assert_eq!(
            db.lock().unwrap().optional_cache_bytes(),
            cached - reclaimed
        );
        assert_eq!(
            db.lock()
                .unwrap()
                .lazy_vector_source
                .as_ref()
                .unwrap()
                .reserved_bytes(),
            directory
        );
        drop((busy, blocker, retained, text, vector));
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory
        );
        drop(db);
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn result_array_reclamation_keeps_fusion_and_hits_charged() {
    let path = cached_vector_batch_fixture("result-array-reclaim");
    let original = fs::read(&path).unwrap();
    let array_bytes = 2 * size_of::<SearchResult>();
    let fused_bytes = 2 * size_of::<CandidateScore<'_>>();
    for (parent, allowed) in [false, true].into_iter().flat_map(|parent| {
        [array_bytes - 1, array_bytes, array_bytes + 1].map(|allowed| (parent, allowed))
    }) {
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let (_, directory, cached) = cache_for_preparation(&db);
        let (text, vector, hits_bytes) = owned_hits(&context);
        let scores = fuse_candidates(
            &db.lock().unwrap().catalog,
            &text,
            &vector,
            None,
            SearchWeights::new(0.5, 0.5, 0.0).unwrap(),
            2,
            &context,
        )
        .unwrap();
        assert_eq!(scores.reserved_bytes(), fused_bytes);
        let pointers = (scores.as_slice().as_ptr(), text.as_ptr(), vector.as_ptr());
        let blocker = parent.then(|| {
            context
                .cache_reservation(
                    (context.available_total_bytes() - allowed) as u128,
                    usize::MAX,
                )
                .unwrap()
        });
        let mut busy = context.workspace_reservation();
        if !parent {
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                - directory
                - hits_bytes
                - fused_bytes
                - allowed;
            busy.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
        }
        let mut attempts = 0;
        let result = db.lock().unwrap().read_reclaiming(|_| {
            attempts += 1;
            let result =
                ReservedVec::<SearchResult>::with_capacity(2, &context, "operation_scratch");
            if attempts == 1 && allowed < array_bytes {
                assert_memory_error(
                    result.as_ref().err().unwrap(),
                    if parent {
                        TOTAL
                    } else {
                        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    },
                );
            }
            result
        });
        assert_eq!(attempts, if allowed < array_bytes { 2 } else { 1 });
        if !parent && allowed < array_bytes {
            let error = result.err().unwrap();
            assert_memory_error(&error, JsonProcessingPolicyV1::WORK_BUDGET_BYTES);
        } else {
            let array = result.unwrap();
            assert!(array.as_slice().is_empty());
            assert_eq!(array.reserved_bytes(), array_bytes);
            assert_eq!(
                context.available_workspace_bytes(),
                allowed
                    + if parent && allowed < array_bytes {
                        cached
                    } else {
                        0
                    }
                    - array_bytes
            );
            drop(array);
        }
        assert_fused(scores.as_slice(), &text);
        assert_eq!(
            pointers,
            (scores.as_slice().as_ptr(), text.as_ptr(), vector.as_ptr())
        );
        assert_eq!(
            db.lock().unwrap().optional_cache_bytes(),
            if allowed < array_bytes { 0 } else { cached }
        );
        assert_eq!(
            context.available_workspace_bytes(),
            allowed
                + if parent && allowed < array_bytes {
                    cached
                } else {
                    0
                }
        );
        drop((busy, blocker));
        drop(scores);
        drop((text, vector));
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory
        );
        drop(db);
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn fusion_and_result_array_native_failures_do_not_reclaim_or_repeat() {
    let path = cached_vector_batch_fixture("query-output-native");
    let original = fs::read(&path).unwrap();
    let db = Database::open(&path).unwrap();
    let context = db.json_context();
    let (_, directory, cached) = cache_for_preparation(&db);
    let (text, vector, _) = owned_hits(&context);
    let baseline = context.available_workspace_bytes();
    for ordinal in [1, 2] {
        let mut attempts = 0;
        let error = crate::test_alloc::fail_nth_allocation_once(ordinal, || {
            db.lock().unwrap().read_reclaiming(|inner| {
                attempts += 1;
                fuse_candidates(
                    &inner.catalog,
                    &text,
                    &vector,
                    None,
                    SearchWeights::new(0.5, 0.5, 0.0).unwrap(),
                    2,
                    &context,
                )
            })
        })
        .err()
        .unwrap();
        assert_eq!(attempts, 1);
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(error.detail().is_empty());
        assert_eq!(context.available_workspace_bytes(), baseline);
        assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached);
    }
    let scores = fuse_candidates(
        &db.lock().unwrap().catalog,
        &text,
        &vector,
        None,
        SearchWeights::new(0.5, 0.5, 0.0).unwrap(),
        2,
        &context,
    )
    .unwrap();
    let baseline = context.available_workspace_bytes();
    let mut attempts = 0;
    let error = crate::test_alloc::fail_nth_allocation_once(1, || {
        db.lock().unwrap().read_reclaiming(|_| {
            attempts += 1;
            ReservedVec::<SearchResult>::with_capacity(2, &context, "operation_scratch")
        })
    })
    .err()
    .unwrap();
    assert_eq!(attempts, 1);
    assert_eq!(error.code(), ErrorCode::IoError);
    assert!(error.detail().is_empty());
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached);
    assert_fused(scores.as_slice(), &text);
    drop(scores);
    drop((text, vector));
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory
    );
    drop(db);
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

pub(super) fn large_query_cache_fixture(name: &str) -> PathBuf {
    let path = cached_vector_batch_fixture(name);
    {
        let db = Database::open(&path).unwrap();
        let cached = db.collection("cached").unwrap();
        cached
            .insert_document(DocumentInput {
                external_id: "extra".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..24)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("extra-{ordinal}"),
                        text: "stable content".into(),
                        ordinal,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        cached
            .put_embeddings(
                (0..24)
                    .map(|ordinal| EmbeddingInput {
                        chunk_external_id: format!("extra-{ordinal}"),
                        model_name: "m".into(),
                        dim: 1024,
                        vector: vec![1.0; 1024],
                    })
                    .collect(),
            )
            .unwrap();
    }
    path
}

#[test]
fn search_reclaims_at_fusion_and_array_preserving_results_and_statistics() {
    let path = large_query_cache_fixture("search-output-reclaim");
    let original = fs::read(&path).unwrap();
    for stage in [QueryOutputStage::Fusion, QueryOutputStage::ResultArray] {
        for top_k in [1, 2] {
            let db = Database::open_read_only(&path).unwrap();
            let handle = db.collection("first").unwrap();
            let vector = QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            };
            let filter = JsonValue::object();
            let context = db.json_context();
            let input = reserve_query_json_inputs(&context, Some(&vector), &filter, None).unwrap();
            assert_eq!(input.reserved_bytes(), 9);
            let (_, directory, cached) = cache_for_preparation(&db);
            assert!(cached > 128 * 1024);
            let signature = db.lock().unwrap().catalog.state_signature();
            let commit = db.lock().unwrap().last_commit_sequence_no;
            let (reference, reference_work) = db
                .search_in_collection(
                    handle.collection_id(),
                    None,
                    Some(&vector),
                    &filter,
                    None,
                    false,
                    None,
                    top_k,
                    None,
                    Some(context.clone()),
                )
                .unwrap();
            let expected = reference.to_json();
            let score_bits = reference
                .results
                .iter()
                .map(|row| row.final_score.to_bits())
                .collect::<Vec<_>>();
            drop((reference, reference_work));
            let collection = db
                .lock()
                .unwrap()
                .catalog
                .collection_by_name("first")
                .unwrap()
                .clone();
            let collection_bytes = collection.collection_id.len()
                + collection.name.len()
                + collection.embedding_model.as_ref().unwrap().len()
                + collection.created_at.len();
            let ids_bytes: usize = db
                .lock()
                .unwrap()
                .catalog
                .live_chunks()
                .filter(|chunk| chunk.collection_id == collection.collection_id)
                .map(|chunk| chunk.chunk_id.len())
                .sum();
            let hits_bytes = 4 * size_of::<VectorHit>() + ids_bytes;
            let fusion_bytes = 2 * size_of::<CandidateScore<'_>>();
            let retained_bytes = directory
                + 9
                + collection_bytes
                + 16
                + 21
                + hits_bytes
                + if stage == QueryOutputStage::ResultArray {
                    fusion_bytes
                } else {
                    0
                };
            let allowed = if stage == QueryOutputStage::Fusion {
                2 * size_of::<usize>() + fusion_bytes - 1
            } else {
                top_k * size_of::<SearchResult>() - 1
            };
            let held = Arc::new(Mutex::new(None));
            let hook_held = Arc::clone(&held);
            let hook_context = context.clone();
            let hook_inner = Arc::downgrade(&db.inner);
            db.lock().unwrap().query_output_hook = Some((
                stage,
                Box::new(move || {
                    assert!(matches!(
                        hook_inner.upgrade().unwrap().try_lock(),
                        Err(std::sync::TryLockError::WouldBlock)
                    ));
                    assert_eq!(
                        hook_context.available_workspace_bytes(),
                        JsonProcessingPolicyV1::WORK_BUDGET_BYTES - retained_bytes
                    );
                    *hook_held.lock().unwrap() = Some(
                        hook_context
                            .cache_reservation(
                                (hook_context.available_total_bytes() - allowed) as u128,
                                usize::MAX,
                            )
                            .unwrap(),
                    );
                }),
            ));
            let (result, work) = db
                .search_in_collection(
                    handle.collection_id(),
                    None,
                    Some(&vector),
                    &filter,
                    None,
                    false,
                    None,
                    top_k,
                    None,
                    Some(context.clone()),
                )
                .unwrap();
            assert_eq!(result.to_json(), expected);
            assert_eq!(
                result
                    .results
                    .iter()
                    .map(|row| row.final_score.to_bits())
                    .collect::<Vec<_>>(),
                score_bits
            );
            assert_eq!(
                result.query_context.execution_stats,
                SearchExecutionStats {
                    vector_scan_count: 2,
                    distance_ops: 4,
                    fusion_candidate_count: 2,
                }
            );
            {
                let inner = db.lock().unwrap();
                assert!(inner.query_output_hook.is_none());
                assert_eq!(inner.optional_cache_bytes(), 0);
                assert_eq!(inner.catalog.state_signature(), signature);
                assert_eq!(inner.last_commit_sequence_no, commit);
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
                assert_eq!(
                    inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
                    directory
                );
            }
            drop((result, work));
            drop(held.lock().unwrap().take());
            assert_eq!(
                context.available_workspace_bytes(),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory - 9
            );
            drop((vector, filter, input, handle, db));
            assert_eq!(fs::read(&path).unwrap(), original);
        }
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn expanded_window_scoring_failure_releases_live_candidates_and_allows_same_handle_retry() {
    let path = large_query_cache_fixture("expanded-fusion-cleanup");
    {
        let writer = Database::open(&path).unwrap();
        let collection = writer.collection("first").unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "expanded".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..80)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("expanded-{ordinal}"),
                        text: "needle hay".into(),
                        ordinal,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(
                (0..80)
                    .map(|ordinal| EmbeddingInput {
                        chunk_external_id: format!("expanded-{ordinal}"),
                        model_name: "m".into(),
                        dim: 2,
                        vector: vec![1.0, (ordinal % 7) as f32],
                    })
                    .collect(),
            )
            .unwrap();
    }
    let original = fs::read(&path).unwrap();
    let db = Database::open_read_only(&path).unwrap();
    let handle = db.collection("first").unwrap();
    let query = || {
        handle
            .search()
            .text("needle")
            .vector(vec![1.0, 0.0])
            .resource_profile(ResourceProfile::Constrained)
            .top_k(600)
            .execute()
    };
    let expected = {
        let result = query().unwrap();
        assert_eq!(result.query_context.candidate_limits.fusion, 600);
        assert!(result.query_context.execution_stats.fusion_candidate_count > 50);
        result.to_json()
    };
    let context = db.json_context();
    let (_, directory, _) = cache_for_preparation(&db);
    let baseline = context.available_workspace_bytes();
    let (signature, commit) = {
        let inner = db.lock().unwrap();
        (
            inner.catalog.state_signature(),
            inner.last_commit_sequence_no,
        )
    };
    let held = Arc::new(Mutex::new(None));
    let hook_held = Arc::clone(&held);
    let hook_context = context.clone();
    db.lock().unwrap().query_output_hook = Some((
        QueryOutputStage::Fusion,
        Box::new(move || {
            let occupied = hook_context.available_workspace_bytes();
            let mut blocker = hook_context.workspace_reservation();
            blocker
                .grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            *hook_held.lock().unwrap() = Some(blocker);
        }),
    ));
    let error = query().unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    assert_eq!(
        error_detail_value(&error, "operation"),
        Some("operation_scratch")
    );
    {
        let inner = db.lock().unwrap();
        assert!(inner.query_output_hook.is_none());
        assert_eq!(inner.optional_cache_bytes(), 0);
        assert_eq!(inner.catalog.state_signature(), signature);
        assert_eq!(inner.last_commit_sequence_no, commit);
        assert_eq!(
            inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
            directory
        );
    }
    assert_eq!(db.temporary_work.used(), 0);
    drop(held.lock().unwrap().take());
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(query().unwrap().to_json(), expected);
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(fs::read(&path).unwrap(), original);
    drop((handle, db));
    fs::remove_file(path).unwrap();
}

#[test]
fn invalid_top_k_is_rejected_before_candidate_or_index_work() {
    let path = cached_vector_batch_fixture("top-k-pre-index");
    let original = fs::read(&path).unwrap();
    let db = Database::open_read_only(&path).unwrap();
    let handle = db.collection("first").unwrap();
    let context = db.json_context();
    let baseline = context.available_workspace_bytes();
    for top_k in [0, usize::MAX] {
        if top_k != 0 && top_k <= crate::query::MAX_SEARCH_TOP_K {
            continue;
        }
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hook_calls = Arc::clone(&calls);
        db.lock().unwrap().query_output_hook = Some((
            QueryOutputStage::Fusion,
            Box::new(move || {
                hook_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }),
        ));
        let error = handle
            .search()
            .text("stable")
            .vector(vec![1.0, 0.0])
            .top_k(top_k)
            .execute()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidSearchParams);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        let mut inner = db.lock().unwrap();
        assert!(inner.lazy_vector_segment.is_none());
        assert!(inner.persistent_vector_segments.is_none());
        assert_eq!(inner.text_index.term_count(), 0);
        assert!(inner.query_output_hook.take().is_some());
        drop(inner);
        assert_eq!(context.available_workspace_bytes(), baseline);
        assert_eq!(db.temporary_work.used(), 0);
    }
    assert!(handle.search().text("stable").top_k(1).execute().is_ok());
    assert_eq!(fs::read(&path).unwrap(), original);
    drop((handle, db));
    fs::remove_file(path).unwrap();
}
