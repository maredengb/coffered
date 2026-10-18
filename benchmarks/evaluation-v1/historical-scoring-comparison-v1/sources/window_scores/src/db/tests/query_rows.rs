use super::query_output::large_query_cache_fixture;
use super::query_preparation::cache_for_preparation;
use super::*;
use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
use std::mem::size_of;

const TOTAL: usize = 3 * 1024 * 1024 * 1024;
const LONG_ID: usize = 8193;
const METADATA: usize = 4097;

fn fixture(name: &str) -> PathBuf {
    let path = large_query_cache_fixture(name);
    let db = Database::open(&path).unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            name: "rows".into(),
            embedding_dim: 2,
            tokenizer: Default::default(),
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    for (id, external_id, value) in [
        ("a", "first".into(), "s".into()),
        ("b", "d".repeat(LONG_ID), "v".repeat(METADATA)),
    ] {
        collection
            .insert_document(DocumentInput {
                external_id,
                title: None,
                source_uri: None,
                metadata: object_from_pairs([("m", JsonValue::String(value))]),
                chunks: vec![ChunkInput {
                    chunk_external_id: id.into(),
                    text: "alpha".into(),
                    ordinal: 0,
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            })
            .unwrap();
        collection
            .put_embeddings(vec![EmbeddingInput {
                chunk_external_id: id.into(),
                model_name: "m".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            }])
            .unwrap();
    }
    drop((collection, db));
    path
}

fn query(
    db: &Database,
    collection: &CollectionInfo,
    context: &JsonOperationContext,
) -> (SearchResults, RecordReservation) {
    db.search_in_collection(
        &collection.collection_id,
        Some("alpha"),
        Some(&QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        }),
        &JsonValue::object(),
        None,
        false,
        None,
        2,
        None,
        Some(context.clone()),
    )
    .unwrap()
}

fn row(
    inner: &Inner,
    collection: &CollectionInfo,
    id: &str,
    query: &SearchQueryContext,
    context: &JsonOperationContext,
    work: &mut RecordReservation,
) -> Result<Option<SearchResult>> {
    build_result(
        inner,
        collection,
        CandidateScore {
            chunk_id: id,
            bm25_raw: Some(1.0),
            bm25_norm: 1.0,
            vector_raw: Some(1.0),
            vector_norm: Some(1.0),
            final_score: 1.0,
            ..Default::default()
        },
        &[],
        None,
        &JsonValue::object(),
        query,
        query.profile,
        query.candidate_limits,
        query.execution_stats,
        Some(1.0),
        context,
        work,
    )
}

fn new_work(context: &JsonOperationContext) -> RecordReservation {
    RecordReservation::new(context.reservation(), context.workspace_reservation())
}

fn json_bytes(value: &JsonValue) -> usize {
    match value {
        JsonValue::String(value) => value.capacity(),
        JsonValue::Array(values) => {
            values.capacity() * size_of::<JsonValue>()
                + values.iter().map(json_bytes).sum::<usize>()
        }
        JsonValue::Object(values) => {
            let nodes = if values.is_empty() {
                0
            } else {
                1 + (values.len() - 1) / 5
            };
            nodes * (11 * size_of::<(String, JsonValue)>() + 16 * size_of::<usize>())
                + values
                    .iter()
                    .map(|(key, value)| key.capacity() + json_bytes(value))
                    .sum::<usize>()
        }
        _ => 0,
    }
}

fn row_bytes(row: &SearchResult) -> (usize, usize) {
    (
        json_bytes(&row.document_metadata)
            + json_bytes(&row.chunk_metadata)
            + json_bytes(&row.explain),
        row.chunk_id.capacity()
            + row.chunk_external_id.capacity()
            + row.document_id.capacity()
            + row.document_external_id.capacity()
            + row.snippet.capacity(),
    )
}

fn chunk_bytes(chunk: &Chunk) -> usize {
    assert_eq!(chunk.chunk_id.len(), 20);
    assert_eq!(chunk.document_id.len(), 20);
    assert_eq!(chunk.collection_id.len(), 20);
    assert_eq!(chunk.text_hash.len(), 64);
    chunk.chunk_id.len()
        + chunk.document_id.len()
        + chunk.collection_id.len()
        + chunk.chunk_external_id.len()
        + chunk.text.len()
        + chunk.text_hash.len()
        + chunk.created_at.len()
        + chunk.embedding_id.as_ref().map_or(0, String::len)
}

#[test]
fn row_reclamation_releases_failed_stages_and_retains_previous_rows() {
    let path = fixture("row-partial-reclaim");
    let original = fs::read(&path).unwrap();
    for parent in [true, false] {
        let db = Database::open_read_only(&path).unwrap();
        let handle = db.collection("rows").unwrap();
        let collection = db
            .lock()
            .unwrap()
            .catalog
            .collection_by_name("rows")
            .unwrap()
            .clone();
        let context = db.json_context();
        let (reference, reference_work) = query(&db, &collection, &context);
        let query_context = reference.query_context.clone();
        drop((reference, reference_work));
        let a = handle.get_chunk_by_external_id("a").unwrap();
        let b = handle.get_chunk_by_external_id("b").unwrap();
        let decoded = chunk_bytes(&b);
        let (catalog, directory, cached) = cache_for_preparation(&db);
        let mut first_work = new_work(&context);
        let first = row(
            &db.lock().unwrap(),
            &collection,
            &a.chunk_id,
            &query_context,
            &context,
            &mut first_work,
        )
        .unwrap()
        .unwrap();
        let first_bytes = row_bytes(&first);
        assert_eq!(
            (first_work.json_bytes(), first_work.workspace_bytes()),
            first_bytes
        );
        let first_pointer = first.explain.as_object().unwrap() as *const _;
        let baseline = context.available_workspace_bytes();
        let allowed = decoded + 2 * METADATA + 4096;
        let blocker = parent.then(|| {
            context
                .cache_reservation(
                    (TOTAL - catalog - directory - cached - first_bytes.0 - first_bytes.1 - allowed)
                        as u128,
                    usize::MAX,
                )
                .unwrap()
        });
        let mut busy = context.reservation();
        if !parent {
            busy.grow(baseline - allowed, crate::json::JsonInputKind::StoredJson)
                .unwrap();
        }
        assert_eq!(context.available_workspace_bytes(), allowed);
        let mut attempts = 0;
        let mut first_error = None;
        let result = db.lock().unwrap().read_reclaiming(|inner| {
            attempts += 1;
            let mut work = new_work(&context);
            let (result, allocations) = crate::test_alloc::key_allocations(METADATA, || {
                row(
                    inner,
                    &collection,
                    &b.chunk_id,
                    &query_context,
                    &context,
                    &mut work,
                )
            });
            if let Err(error) = &result {
                assert!(work.json_bytes() > 0);
                assert!(work.json_bytes() <= allowed);
                assert_eq!(allocations.allocated, 1);
                assert_eq!(allocations.live, 0);
                assert!(!allocations.overflow);
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                assert_eq!(
                    error_detail_value(error, "json_input_kind"),
                    Some("stored_json")
                );
                assert_eq!(
                    error_detail_value(error, "json_limit_reason"),
                    Some("work_budget_exceeded")
                );
                let limit = error_detail_value(error, "limit")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                let requested = error_detail_value(error, "actual")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                assert!(requested > limit);
                if let Some(previous) = &first_error {
                    assert_eq!(error.detail(), previous);
                } else {
                    first_error = Some(error.detail().to_vec());
                }
            }
            Ok((result?, work))
        });
        assert_eq!(attempts, 2);
        assert_eq!(db.lock().unwrap().optional_cache_bytes(), 0);
        if parent {
            let (second, work) = result.unwrap();
            let second = second.unwrap();
            assert_eq!(second.document_external_id, "d".repeat(LONG_ID));
            assert_eq!(
                second
                    .document_metadata
                    .get("m")
                    .and_then(JsonValue::as_str),
                Some("v".repeat(METADATA).as_str())
            );
            assert_eq!(
                (work.json_bytes(), work.workspace_bytes()),
                row_bytes(&second)
            );
            drop((second, work));
            assert_eq!(context.available_workspace_bytes(), allowed + cached);
        } else {
            assert_eq!(result.err().unwrap().detail(), first_error.unwrap());
            assert_eq!(context.available_workspace_bytes(), allowed);
        }
        assert_eq!(first.document_external_id, "first");
        assert_eq!(
            first.explain.as_object().unwrap() as *const _,
            first_pointer
        );
        assert_eq!(
            (first_work.json_bytes(), first_work.workspace_bytes()),
            first_bytes
        );
        drop((busy, blocker, first, first_work));
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory
        );
        drop((handle, db));
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn row_native_failure_damage_and_absent_rows_do_not_reclaim_cache() {
    let path = fixture("row-native-damage");
    let original = fs::read(&path).unwrap();
    let db = Database::open_read_only(&path).unwrap();
    let handle = db.collection("rows").unwrap();
    let collection = db
        .lock()
        .unwrap()
        .catalog
        .collection_by_name("rows")
        .unwrap()
        .clone();
    let context = db.json_context();
    let (reference, reference_work) = query(&db, &collection, &context);
    let query_context = reference.query_context.clone();
    drop((reference, reference_work));
    let b = handle.get_chunk_by_external_id("b").unwrap();
    let (_, _, cached) = cache_for_preparation(&db);
    let baseline = context.available_workspace_bytes();
    let mut attempts = 0;
    let error = crate::test_alloc::fail_byte_allocation_once(LONG_ID, || {
        db.lock().unwrap().read_reclaiming(|inner| {
            attempts += 1;
            let mut work = new_work(&context);
            row(
                inner,
                &collection,
                &b.chunk_id,
                &query_context,
                &context,
                &mut work,
            )
        })
    })
    .unwrap_err();
    assert_eq!(attempts, 1);
    assert_eq!(error.code(), ErrorCode::IoError);
    assert!(error.detail().is_empty());
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached);
    let offset = db
        .lock()
        .unwrap()
        .catalog
        .record_ref_for_chunk(&b.chunk_id)
        .unwrap()
        .payload
        .offset;
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&[original[offset as usize] ^ 1]).unwrap();
    attempts = 0;
    let error = db
        .lock()
        .unwrap()
        .read_reclaiming(|inner| {
            attempts += 1;
            let mut work = new_work(&context);
            row(
                inner,
                &collection,
                &b.chunk_id,
                &query_context,
                &context,
                &mut work,
            )
        })
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
    assert_eq!(attempts, 1);
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached);
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&[original[offset as usize]]).unwrap();
    drop(file);
    attempts = 0;
    let result = db
        .lock()
        .unwrap()
        .read_reclaiming(|inner| {
            attempts += 1;
            let mut work = new_work(&context);
            let result = row(
                inner,
                &collection,
                "missing",
                &query_context,
                &context,
                &mut work,
            )?;
            assert_eq!((work.json_bytes(), work.workspace_bytes()), (0, 0));
            Ok(result)
        })
        .unwrap();
    assert!(result.is_none());
    assert_eq!(attempts, 1);
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached);
    drop((handle, db));
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn complete_row_peak_has_exact_parent_and_working_boundaries() {
    let path = fixture("row-peak-boundaries");
    let original = fs::read(&path).unwrap();
    for (parent, delta) in [false, true]
        .into_iter()
        .flat_map(|parent| [-1isize, 0, 1].map(|delta| (parent, delta)))
    {
        let db = Database::open_read_only(&path).unwrap();
        let handle = db.collection("rows").unwrap();
        let collection = db
            .lock()
            .unwrap()
            .catalog
            .collection_by_name("rows")
            .unwrap()
            .clone();
        let context = db.json_context();
        let (reference, reference_work) = query(&db, &collection, &context);
        let query_context = reference.query_context.clone();
        drop((reference, reference_work));
        let b = handle.get_chunk_by_external_id("b").unwrap();
        let mut reference_work = new_work(&context);
        let reference = row(
            &db.lock().unwrap(),
            &collection,
            &b.chunk_id,
            &query_context,
            &context,
            &mut reference_work,
        )
        .unwrap()
        .unwrap();
        let retained = row_bytes(&reference);
        assert_eq!(
            (
                reference_work.json_bytes(),
                reference_work.workspace_bytes()
            ),
            retained
        );
        let expected = reference.to_json();
        // The decoded chunk still coexists with the completed explain and the
        // three final wrapper strings until build_result drops that chunk.
        let peak = retained.0 + chunk_bytes(&b) + 20 + LONG_ID + 5;
        drop((reference, reference_work));
        let (catalog, directory, cached) = cache_for_preparation(&db);
        let allowed = (peak as isize + delta) as usize;
        let blocker = parent.then(|| {
            context
                .cache_reservation(
                    (TOTAL - catalog - directory - cached - allowed) as u128,
                    usize::MAX,
                )
                .unwrap()
        });
        let mut busy = context.workspace_reservation();
        if !parent {
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory - allowed;
            busy.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
        }
        assert_eq!(context.available_workspace_bytes(), allowed);
        let mut attempts = 0;
        let result = db.lock().unwrap().read_reclaiming(|inner| {
            attempts += 1;
            let mut work = new_work(&context);
            let result = row(
                inner,
                &collection,
                &b.chunk_id,
                &query_context,
                &context,
                &mut work,
            );
            if delta == -1 && attempts == 1 {
                assert_eq!(work.json_bytes(), retained.0);
                let error = result.as_ref().err().unwrap();
                let limit = if parent {
                    TOTAL
                } else {
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                };
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
            Ok((result?, work))
        });
        assert_eq!(attempts, if delta == -1 { 2 } else { 1 });
        if !parent && delta == -1 {
            assert_eq!(result.err().unwrap().code(), ErrorCode::ResourceExhausted);
            assert_eq!(context.available_workspace_bytes(), allowed);
        } else {
            let (result, work) = result.unwrap();
            let result = result.unwrap();
            assert_eq!(result.to_json(), expected);
            assert_eq!((work.json_bytes(), work.workspace_bytes()), retained);
            drop((result, work));
            assert_eq!(
                context.available_workspace_bytes(),
                allowed + if delta == -1 { cached } else { 0 }
            );
        }
        assert_eq!(
            db.lock().unwrap().optional_cache_bytes(),
            if delta == -1 { 0 } else { cached }
        );
        drop((busy, blocker));
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory
        );
        drop((handle, db));
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn hybrid_search_reclaims_second_row_without_replaying_retrieval() {
    let path = fixture("hybrid-row-reclaim");
    let original = fs::read(&path).unwrap();
    let db = Database::open_read_only(&path).unwrap();
    let handle = db.collection("rows").unwrap();
    let collection = db
        .lock()
        .unwrap()
        .catalog
        .collection_by_name("rows")
        .unwrap()
        .clone();
    let context = db.json_context();
    let vector = QueryEmbeddingInput {
        model_name: Some("m".into()),
        dim: 2,
        vector: vec![1.0, 0.0],
    };
    let filter = JsonValue::object();
    let input = reserve_query_json_inputs(&context, Some(&vector), &filter, None).unwrap();
    assert_eq!(input.reserved_bytes(), 9);
    let (catalog, directory, cached) = cache_for_preparation(&db);
    let (reference, reference_work) = query(&db, &collection, &context);
    assert_eq!(reference.results[0].chunk_external_id, "a");
    assert_eq!(reference.results[1].chunk_external_id, "b");
    let expected = reference.to_json();
    let scores = reference
        .results
        .iter()
        .map(|row| row.final_score.to_bits())
        .collect::<Vec<_>>();
    let first = row_bytes(&reference.results[0]);
    drop((reference, reference_work));
    let b = handle.get_chunk_by_external_id("b").unwrap();
    let allowed = chunk_bytes(&b) + 2 * METADATA + 4096;
    let collection_bytes = collection.collection_id.len()
        + collection.name.len()
        + collection.embedding_model.as_ref().unwrap().len()
        + collection.created_at.len();
    let hits = 4 * size_of::<VectorHit>()
        + 40
        + 2 * size_of::<Bm25Hit>()
        + 40
        + 2 * (size_of::<MatchedTerm>() + 5);
    let retained = directory
        + 9
        + collection_bytes
        + 16
        + 21
        + hits
        + 2 * size_of::<CandidateScore<'_>>()
        + 2 * size_of::<SearchResult>()
        + first.0
        + first.1;
    let signature = db.lock().unwrap().catalog.state_signature();
    let commit = db.lock().unwrap().last_commit_sequence_no;
    let held = Arc::new(Mutex::new(None));
    let hook_held = Arc::clone(&held);
    let hook_context = context.clone();
    let weak = Arc::downgrade(&db.inner);
    db.lock().unwrap().query_output_hook = Some((
        QueryOutputStage::ResultRow(1),
        Box::new(move || {
            assert!(matches!(
                weak.upgrade().unwrap().try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            assert_eq!(
                hook_context.available_workspace_bytes(),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - retained
            );
            *hook_held.lock().unwrap() = Some(
                hook_context
                    .cache_reservation(
                        (TOTAL - catalog - cached - retained - allowed) as u128,
                        usize::MAX,
                    )
                    .unwrap(),
            );
        }),
    ));
    let (actual, work) = query(&db, &collection, &context);
    assert_eq!(actual.to_json(), expected);
    let row_json: usize = actual.results.iter().map(|row| row_bytes(row).0).sum();
    let row_workspace: usize = actual.results.iter().map(|row| row_bytes(row).1).sum();
    assert_eq!(work.json_bytes(), row_json);
    assert_eq!(
        work.workspace_bytes(),
        row_workspace + 2 * size_of::<SearchResult>() + 16 + 21 + collection.name.len() + 3 + 5
    );
    assert_eq!(
        actual
            .results
            .iter()
            .map(|row| row.final_score.to_bits())
            .collect::<Vec<_>>(),
        scores
    );
    assert_eq!(
        actual.query_context.execution_stats,
        SearchExecutionStats {
            vector_scan_count: 2,
            distance_ops: 4,
            fusion_candidate_count: 2
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
    drop((actual, work));
    drop(held.lock().unwrap().take());
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory - 9
    );
    drop((input, handle, db));
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}
