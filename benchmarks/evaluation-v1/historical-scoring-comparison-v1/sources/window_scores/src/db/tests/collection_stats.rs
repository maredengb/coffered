use super::query_output::large_query_cache_fixture;
use super::query_preparation::cache_for_preparation;
use super::*;
use crate::json::JsonProcessingPolicyV1;

const TOTAL: usize = 3 * 1024 * 1024 * 1024;
const NAME_BYTES: usize = 65536;

fn fixture(label: &str) -> PathBuf {
    let path = large_query_cache_fixture(label);
    let db = Database::open(&path).unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            name: "n".repeat(NAME_BYTES),
            embedding_dim: 2,
            tokenizer: Default::default(),
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "stats-document".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "stats-chunk".into(),
                text: "alpha beta gamma".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
    collection
        .put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "stats-chunk".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
    path
}

#[test]
fn stats_name_reclaims_only_parent_pressure_at_exact_copy_boundaries() {
    let path = fixture("stats-name-boundaries");
    let original = fs::read(&path).unwrap();
    let name = "n".repeat(NAME_BYTES);
    for (parent, allowed) in [false, true].into_iter().flat_map(|parent| {
        [NAME_BYTES - 1, NAME_BYTES, NAME_BYTES + 1].map(|allowed| (parent, allowed))
    }) {
        let db = Database::open_read_only(&path).unwrap();
        let handle = db.collection(&name).unwrap();
        let context = db.json_context();
        let (catalog, directory, cached) = cache_for_preparation(&db);
        assert!(cached > 128 * 1024);
        let stats = handle.stats().unwrap();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.chunk_count, 1);
        assert_eq!(stats.embedding_count, 1);
        assert_eq!(stats.vector_count, 1);
        assert_eq!(stats.text_index_terms, 3);
        let reference = stats.to_json();
        drop(stats);
        let (signature, commit) = {
            let inner = db.lock().unwrap();
            (
                inner.catalog.state_signature(),
                inner.last_commit_sequence_no,
            )
        };
        let other = db.json_context();
        let mut retained = other.workspace_reservation();
        retained.grow_to(37, 37, "operation_scratch").unwrap();
        let blocker = parent.then(|| {
            context
                .cache_reservation(
                    (TOTAL - catalog - directory - cached - 37 - allowed) as u128,
                    usize::MAX,
                )
                .unwrap()
        });
        let mut busy = other.workspace_reservation();
        if !parent {
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory - 37 - allowed;
            busy.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
        }
        assert_eq!(context.available_workspace_bytes(), allowed);
        let (outcome, allocations) =
            crate::test_alloc::key_allocations(NAME_BYTES, || handle.stats());
        if !parent && allowed < NAME_BYTES {
            let error = outcome.err().unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("operation_scratch")
            );
            assert_eq!(
                error_detail_value(&error, "limit_bytes"),
                Some(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                        .to_string()
                        .as_str()
                )
            );
            assert_eq!(
                error_detail_value(&error, "requested_bytes"),
                Some(
                    (JsonProcessingPolicyV1::WORK_BUDGET_BYTES + 1)
                        .to_string()
                        .as_str()
                )
            );
            assert_eq!(allocations.allocated, 0);
            assert_eq!(allocations.live, 0);
        } else {
            let stats = outcome.unwrap();
            assert_eq!(stats.collection, name);
            assert_eq!(stats.collection.capacity(), NAME_BYTES);
            assert_eq!(stats.to_json(), reference);
            assert_eq!(allocations.allocated, 1);
            assert_eq!(allocations.live, 1);
            assert_eq!(allocations.peak, 1);
        }
        assert!(!allocations.overflow);
        let reclaimed = if allowed < NAME_BYTES { cached } else { 0 };
        assert_eq!(
            context.available_workspace_bytes(),
            allowed + if parent { reclaimed } else { 0 }
        );
        assert_eq!(retained.reserved_bytes(), 37);
        {
            let inner = db.lock().unwrap();
            assert_eq!(inner.optional_cache_bytes(), cached - reclaimed);
            assert_eq!(inner.catalog.reserved_bytes(), catalog);
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
        drop((busy, blocker, retained));
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory
        );
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn stats_native_failure_and_missing_collection_preserve_cache_and_credit() {
    let path = fixture("stats-name-errors");
    let original = fs::read(&path).unwrap();
    let db = Database::open_read_only(&path).unwrap();
    let handle = db.collection(&"n".repeat(NAME_BYTES)).unwrap();
    let context = db.json_context();
    let (_, directory, cached) = cache_for_preparation(&db);
    let baseline = context.available_workspace_bytes();
    let signature = db.lock().unwrap().catalog.state_signature();
    let error = crate::test_alloc::fail_byte_allocation_once(NAME_BYTES, || handle.stats())
        .err()
        .unwrap();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert!(error.detail().is_empty());
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached);
    let (error, allocations) = crate::test_alloc::key_allocations(NAME_BYTES, || {
        db.collection_stats_by_id("missing").err().unwrap()
    });
    assert_eq!(error.code(), ErrorCode::CollectionNotFound);
    assert_eq!(allocations.allocated, 0);
    assert_eq!(context.available_workspace_bytes(), baseline);
    assert_eq!(db.lock().unwrap().optional_cache_bytes(), cached);
    assert_eq!(db.lock().unwrap().catalog.state_signature(), signature);
    assert_eq!(
        baseline,
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES - directory
    );
    drop((handle, db));
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}
