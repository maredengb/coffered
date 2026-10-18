use super::*;
use crate::json::{JsonInputKind, JsonProcessingPolicyV1};
use std::mem::size_of;

const TOTAL: usize = 3 * 1024 * 1024 * 1024;
const LITERAL_BYTES: usize = 8193;

fn filter_input() -> JsonValue {
    object_from_pairs([("chunk.lang", JsonValue::String("x".repeat(LITERAL_BYTES)))])
}

fn compile_peak() -> usize {
    // Independent layout oracle for the four initial compiler frames and one
    // minimum node slot; the two retained strings remain live at that peak.
    #[allow(dead_code)]
    struct Frame<'a> {
        fields: std::collections::btree_map::Iter<'a, String, JsonValue>,
        and_items: Option<std::slice::Iter<'a, JsonValue>>,
        nodes: Vec<FilterExpr>,
    }
    4 * size_of::<Frame<'_>>() + 10 + LITERAL_BYTES + size_of::<FilterExpr>()
}

pub(super) fn cache_for_preparation(db: &Database) -> (usize, usize, usize) {
    let mut inner = db.lock().unwrap();
    let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
    assert!(cached.reserved_bytes() > 32 * 1024);
    inner.persistent_vector_segments = Some(cached);
    (
        inner.catalog.reserved_bytes(),
        inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
        inner.optional_cache_bytes(),
    )
}

fn assert_filter(expr: Option<FilterExpr>) {
    let Some(FilterExpr::Eq(field, JsonValue::String(value))) = &expr else {
        panic!("complete equality filter required");
    };
    assert_eq!(field, "chunk.lang");
    assert_eq!(value, &"x".repeat(LITERAL_BYTES));
}

#[test]
fn filter_preparation_reclaims_at_exact_parent_and_working_peak_boundaries() {
    let path = cached_vector_batch_fixture("filter-prepare-boundaries");
    let original = fs::read(&path).unwrap();
    for (parent, allowed) in [false, true].into_iter().flat_map(|parent| {
        [compile_peak() - 1, compile_peak(), compile_peak() + 1].map(|allowed| (parent, allowed))
    }) {
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let input = filter_input();
        let (key, value) = input.as_object().unwrap().iter().next().unwrap();
        assert_eq!(key.capacity(), 10);
        let JsonValue::String(value) = value else {
            unreachable!()
        };
        assert_eq!(value.capacity(), LITERAL_BYTES);
        let (input_work, _) = input.guard_in(JsonInputKind::Filter, &context).unwrap();
        let input_bytes =
            11 * size_of::<(String, JsonValue)>() + 16 * size_of::<usize>() + 10 + LITERAL_BYTES;
        assert_eq!(input_work.reserved_bytes(), input_bytes);
        let (catalog, directory, cached) = cache_for_preparation(&db);
        let signature = db.lock().unwrap().catalog.state_signature();
        let (collection, collection_work) = copy_collection_in(
            db.lock()
                .unwrap()
                .catalog
                .collection_by_name("first")
                .unwrap(),
            &context,
        )
        .unwrap();
        let collection_bytes = collection.collection_id.len()
            + collection.name.len()
            + collection.embedding_model.as_ref().unwrap().len()
            + collection.created_at.len();
        assert_eq!(collection_work.workspace_bytes(), collection_bytes);
        let other = db.json_context();
        let mut retained = other.workspace_reservation();
        retained.grow_to(37, 37, "operation_scratch").unwrap();
        let blocker = parent.then(|| {
            context
                .cache_reservation(
                    (TOTAL
                        - catalog
                        - directory
                        - cached
                        - input_bytes
                        - collection_bytes
                        - 37
                        - allowed) as u128,
                    usize::MAX,
                )
                .unwrap()
        });
        let mut busy = other.workspace_reservation();
        if !parent {
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                - directory
                - input_bytes
                - collection_bytes
                - 37
                - allowed;
            busy.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
        }
        assert_eq!(context.available_workspace_bytes(), allowed);
        let first = FilterExpr::parse_validated_in(&input, &context, JsonInputKind::Filter);
        if allowed < compile_peak() {
            let error = first.err().unwrap();
            assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
            assert_eq!(
                error_detail_value(&error, "json_limit_reason"),
                Some("work_budget_exceeded")
            );
            assert_eq!(
                error_detail_value(&error, "limit"),
                Some((input_bytes + allowed).to_string().as_str())
            );
            assert_eq!(
                error_detail_value(&error, "actual"),
                Some((input_bytes + compile_peak()).to_string().as_str())
            );
        } else {
            let (expr, work) = first.unwrap();
            assert_filter(expr);
            assert_eq!(work.reserved_bytes(), 10 + LITERAL_BYTES);
            drop(work);
        }
        assert_eq!(context.available_workspace_bytes(), allowed);
        let mut attempts = 0;
        let outcome = db.lock().unwrap().read_reclaiming(|_| {
            attempts += 1;
            FilterExpr::parse_validated_in(&input, &context, JsonInputKind::Filter)
        });
        assert_eq!(attempts, if allowed < compile_peak() { 2 } else { 1 });
        if !parent && allowed < compile_peak() {
            let error = outcome.err().unwrap();
            assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
            assert_eq!(
                error_detail_value(&error, "json_input_kind"),
                Some("filter")
            );
            assert_eq!(
                error_detail_value(&error, "json_limit_reason"),
                Some("work_budget_exceeded")
            );
            assert_eq!(
                error_detail_value(&error, "limit"),
                Some((input_bytes + allowed).to_string().as_str())
            );
            assert_eq!(
                error_detail_value(&error, "actual"),
                Some((input_bytes + compile_peak()).to_string().as_str())
            );
        } else {
            let (expr, work) = outcome.unwrap();
            assert_filter(expr);
            assert_eq!(work.reserved_bytes(), 10 + LITERAL_BYTES);
            drop(work);
        }
        let reclaimed = if allowed < compile_peak() { cached } else { 0 };
        assert_eq!(
            context.available_workspace_bytes(),
            allowed + if parent { reclaimed } else { 0 }
        );
        assert_eq!(input_work.reserved_bytes(), input_bytes);
        assert_eq!(collection_work.workspace_bytes(), collection_bytes);
        assert_eq!(retained.reserved_bytes(), 37);
        {
            let inner = db.lock().unwrap();
            assert_eq!(inner.optional_cache_bytes(), cached - reclaimed);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(
                inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
                directory
            );
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
        }
        assert_eq!(input, filter_input());
        drop((
            collection,
            collection_work,
            input,
            input_work,
            retained,
            busy,
            blocker,
        ));
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
fn filter_preparation_preserves_working_quota_and_native_errors() {
    let path = cached_vector_batch_fixture("filter-prepare-errors");
    let original = fs::read(&path).unwrap();
    let db = Database::open(&path).unwrap();
    let context = db.json_context();
    let input = filter_input();
    let (input_work, _) = input.guard_in(JsonInputKind::Filter, &context).unwrap();
    let (_, directory, cached) = cache_for_preparation(&db);
    let baseline = context.available_workspace_bytes();
    let mut attempts = 0;
    let error = crate::test_alloc::fail_byte_allocation_once(LITERAL_BYTES, || {
        db.lock().unwrap().read_reclaiming(|_| {
            attempts += 1;
            FilterExpr::parse_validated_in(&input, &context, JsonInputKind::Filter)
        })
    })
    .err()
    .unwrap();
    assert_eq!(attempts, 1);
    assert_eq!(error.code(), ErrorCode::IoError);
    assert!(error.detail().is_empty());
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached);
    let other = db.json_context();
    let mut busy = other.workspace_reservation();
    let occupied = baseline - (compile_peak() - 1);
    busy.grow_to(occupied, occupied, "operation_scratch")
        .unwrap();
    let expected = FilterExpr::parse_validated_in(&input, &context, JsonInputKind::Filter)
        .err()
        .unwrap();
    attempts = 0;
    let actual = db
        .lock()
        .unwrap()
        .read_reclaiming(|_| {
            attempts += 1;
            FilterExpr::parse_validated_in(&input, &context, JsonInputKind::Filter)
        })
        .err()
        .unwrap();
    assert_eq!(attempts, 2);
    assert_eq!(actual.code(), ErrorCode::InvalidMetadataFilter);
    assert_eq!(actual.detail(), expected.detail());
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(context.available_workspace_bytes(), compile_peak() - 1);
    assert_eq!(busy.reserved_bytes(), occupied);
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), 0);
    assert_eq!(
        db.lock()
            .unwrap()
            .lazy_vector_source
            .as_ref()
            .unwrap()
            .reserved_bytes(),
        directory
    );
    drop(busy);
    assert_eq!(context.available_workspace_bytes(), baseline);
    let (expr, work) =
        FilterExpr::parse_validated_in(&input, &context, JsonInputKind::Filter).unwrap();
    assert_filter(expr);
    drop((work, input, input_work, db));
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn search_reclaims_optional_cache_for_filter_preparation() {
    let path = cached_vector_batch_fixture("search-prepare-reclaim");
    let original = fs::read(&path).unwrap();
    let db = Database::open(&path).unwrap();
    let handle = db.collection("first").unwrap();
    let context = db.json_context();
    let input = filter_input();
    let (input_work, _) = input.guard_in(JsonInputKind::Filter, &context).unwrap();
    let (catalog, directory, cached) = cache_for_preparation(&db);
    let (collection, work) = copy_collection_in(
        db.lock()
            .unwrap()
            .catalog
            .collection_by_id(handle.collection_id())
            .unwrap(),
        &context,
    )
    .unwrap();
    let collection_bytes = collection.collection_id.len()
        + collection.name.len()
        + collection.embedding_model.as_ref().unwrap().len()
        + collection.created_at.len();
    assert_eq!(work.workspace_bytes(), collection_bytes);
    drop((collection, work));
    let blocker = context
        .cache_reservation(
            (TOTAL
                - catalog
                - directory
                - cached
                - input_work.reserved_bytes()
                - collection_bytes
                - (compile_peak() - 1)) as u128,
            usize::MAX,
        )
        .unwrap();
    let (result, result_work) = db
        .search_in_collection(
            handle.collection_id(),
            Some("the and"),
            None,
            &input,
            None,
            false,
            None,
            1,
            None,
            Some(context.clone()),
        )
        .unwrap();
    assert!(result.results.is_empty());
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), 0);
    assert_eq!(
        db.lock()
            .unwrap()
            .lazy_vector_source
            .as_ref()
            .unwrap()
            .reserved_bytes(),
        directory
    );
    drop((result, result_work, blocker));
    assert_eq!(input, filter_input());
    drop((input, input_work, handle, db));
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn collection_and_why_not_preparation_reclaim_at_the_actual_call_sites() {
    let path = cached_vector_batch_fixture("query-prepare-call-sites");
    {
        let db = Database::open(&path).unwrap();
        db.create_collection(CollectionConfig {
            name: "wide".into(),
            embedding_dim: 2,
            tokenizer: Default::default(),
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".repeat(LITERAL_BYTES)),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    }
    let original = fs::read(&path).unwrap();
    for (why_not, compile_filter) in [(false, false), (true, false), (true, true)] {
        let db = Database::open(&path).unwrap();
        let name = if compile_filter { "first" } else { "wide" };
        let handle = db.collection(name).unwrap();
        let context = db.json_context();
        let input = if compile_filter {
            filter_input()
        } else {
            JsonValue::object()
        };
        let (input_work, _) = input.guard_in(JsonInputKind::Filter, &context).unwrap();
        let (catalog, directory, cached) = cache_for_preparation(&db);
        let signature = db.lock().unwrap().catalog.state_signature();
        let commit = db.lock().unwrap().last_commit_sequence_no;
        let (collection, work) = copy_collection_in(
            db.lock()
                .unwrap()
                .catalog
                .collection_by_id(handle.collection_id())
                .unwrap(),
            &context,
        )
        .unwrap();
        let collection_bytes = collection.collection_id.len()
            + collection.name.len()
            + collection.embedding_model.as_ref().unwrap().len()
            + collection.created_at.len();
        assert_eq!(work.workspace_bytes(), collection_bytes);
        drop((collection, work));
        let expected = if why_not {
            Some(
                db.why_not_debug(
                    name,
                    "absent",
                    Some("the and".into()),
                    None,
                    input.clone(),
                    None,
                    false,
                    1,
                    None,
                )
                .unwrap(),
            )
        } else {
            None
        };
        let available = if compile_filter {
            collection_bytes + compile_peak() - 1
        } else {
            collection_bytes - 1
        };
        let held = Arc::new(Mutex::new(None));
        if why_not {
            let hook_inner = Arc::downgrade(&db.inner);
            let hook_json = Arc::clone(&db.json_work);
            let hook_disk = Arc::clone(&db.temporary_work);
            let hook_context = context.clone();
            let hook_held = Arc::clone(&held);
            let input_bytes = input_work.reserved_bytes();
            let wrapper_bytes = 3 + name.len() + 7 + 16 + 21;
            db.lock().unwrap().before_why_not_ranks = Some(Box::new(move || {
                let hook_db = Database {
                    inner: hook_inner.upgrade().unwrap(),
                    json_work: hook_json,
                    temporary_work: hook_disk,
                };
                assert_eq!(hook_db.lock().unwrap().optional_cache_bytes(), cached);
                assert_eq!(
                    hook_context.available_workspace_bytes(),
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                        - directory
                        - input_bytes
                        - wrapper_bytes
                );
                *hook_held.lock().unwrap() = Some(
                    hook_context
                        .cache_reservation(
                            (TOTAL
                                - catalog
                                - directory
                                - cached
                                - input_bytes
                                - wrapper_bytes
                                - available) as u128,
                            usize::MAX,
                        )
                        .unwrap(),
                );
            }));
        } else {
            *held.lock().unwrap() = Some(
                context
                    .cache_reservation(
                        (TOTAL
                            - catalog
                            - directory
                            - cached
                            - input_work.reserved_bytes()
                            - available) as u128,
                        usize::MAX,
                    )
                    .unwrap(),
            );
        }
        if why_not {
            let (diagnosis, work) = db
                .why_not_debug_with_metadata_boost_in(
                    name,
                    "absent",
                    Some("the and".into()),
                    None,
                    input.clone(),
                    None,
                    false,
                    None,
                    1,
                    None,
                    Some(context.clone()),
                )
                .unwrap();
            assert_eq!(diagnosis.result, expected.as_ref().unwrap().result);
            assert_eq!(diagnosis.query_context.execution_stats.vector_scan_count, 0);
            assert_eq!(
                diagnosis
                    .query_context
                    .execution_stats
                    .fusion_candidate_count,
                0
            );
            drop((diagnosis, work));
        } else {
            let (result, work) = db
                .search_in_collection(
                    handle.collection_id(),
                    Some("the and"),
                    None,
                    &input,
                    None,
                    false,
                    None,
                    1,
                    None,
                    Some(context.clone()),
                )
                .unwrap();
            assert!(result.results.is_empty());
            assert_eq!(result.query_context.execution_stats.vector_scan_count, 0);
            assert_eq!(
                result.query_context.execution_stats.fusion_candidate_count,
                0
            );
            drop((result, work));
        }
        {
            let inner = db.lock().unwrap();
            assert_eq!(inner.optional_cache_bytes(), 0);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.last_commit_sequence_no, commit);
            assert_eq!(
                inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
                directory
            );
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
        }
        drop(held.lock().unwrap().take());
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory - input_work.reserved_bytes(),
            "why_not={why_not}, compile_filter={compile_filter}"
        );
        drop((input, input_work, handle, db));
        assert_eq!(fs::read(&path).unwrap(), original);
    }
    fs::remove_file(path).unwrap();
}
