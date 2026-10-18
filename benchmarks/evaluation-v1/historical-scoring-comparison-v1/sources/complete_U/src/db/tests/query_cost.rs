use super::*;
use crate::search_cost::{Count, Counter, SearchCostTrace, Stage};

#[test]
fn search_cost_cross_window_reuse_and_reads_cover_resident_persistent_and_snapshot_sources() {
    let path = temp_path("query-cost-cross-window");
    {
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "costs".into(),
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
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..51)
                    .map(|id| ChunkInput {
                        chunk_external_id: format!("doc#{id}"),
                        ordinal: id,
                        text: if id == 50 {
                            "needle filler".into()
                        } else {
                            "needle".into()
                        },
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(
                (0..51)
                    .map(|id| EmbeddingInput {
                        chunk_external_id: format!("doc#{id}"),
                        model_name: "m".into(),
                        dim: 2,
                        vector: if id == 50 {
                            vec![1.0, 0.0]
                        } else {
                            vec![-1.0, 0.0]
                        },
                    })
                    .collect(),
            )
            .unwrap();
        db.close().unwrap();
    }
    let original = fs::read(&path).unwrap();
    for source in 0..3 {
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                cache_budget_mb: Some(if source == 1 { 0 } else { 16 }),
                ..Default::default()
            },
        )
        .unwrap();
        let id = db.collection("costs").unwrap().collection_id().to_owned();
        if source == 2 {
            db.lock().unwrap().mark_index_checkpoint_rebuild_needed();
        }
        let embedding = QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        };
        for (query, weights) in [
            ("needle", None),
            ("absent", Some(SearchWeights::new(0.0, 1.0, 0.0).unwrap())),
        ] {
            let context = db.json_context();
            let available = context.available_workspace_bytes();
            let trace = SearchCostTrace::new();
            let (results, retained) = db
                .search_in_collection_traced(
                    &id,
                    Some(query),
                    Some(&embedding),
                    &JsonValue::object(),
                    weights,
                    false,
                    None,
                    1,
                    Some(ResourceProfile::ResourceFirst),
                    Some(context.clone()),
                    Some(&trace),
                )
                .unwrap();
            assert!(trace.valid());
            assert_eq!(
                trace.total(Counter::VectorChecks),
                results.query_context.execution_stats.vector_scan_count as u64
            );
            assert_eq!(
                trace.total(Counter::DistanceComponents),
                results.query_context.execution_stats.distance_ops as u64
            );
            assert_eq!(trace.value(Stage::Recall, Counter::VectorChecks), 51);
            assert_eq!(trace.value(Stage::Recall, Counter::DistanceComponents), 102);
            let completion = u64::from(query == "needle");
            assert_eq!(
                trace.value(Stage::ScoreCompletion, Counter::VectorChecks),
                completion
            );
            assert_eq!(
                trace.value(Stage::ScoreCompletion, Counter::DistanceComponents),
                2 * completion
            );
            assert_eq!(
                trace.value(Stage::NormalizationFusion, Counter::VectorScoreReuse),
                50
            );
            assert_eq!(
                trace.value(Stage::NormalizationFusion, Counter::Bm25ScoreReuse),
                if query == "needle" { 50 } else { 0 }
            );
            assert_eq!(
                trace.value(Stage::ScoreCompletion, Counter::ZeroMatchProofReuse),
                if query == "needle" { 0 } else { 50 }
            );
            let union = if query == "needle" { 51 } else { 50 };
            assert_eq!(
                trace.value(Stage::NormalizationFusion, Counter::FusionCandidateVisits),
                union
            );
            assert!(trace.value(Stage::ResultMaterialization, Counter::PrimaryReadBytes) > 0);
            assert_eq!(
                trace.value(Stage::ResultMaterialization, Counter::PrimaryReadBytes),
                trace.value(Stage::ResultMaterialization, Counter::PrimaryRequestedBytes)
            );
            let json = trace.json(&JsonBuilder::host()).unwrap();
            let counts = json.get("candidate_counts").unwrap();
            assert_eq!(
                counts.get("recall_union_count").unwrap().as_f64(),
                Some(union as f64)
            );
            assert_eq!(
                counts.get("fusion_retained_count").unwrap().as_f64(),
                Some(50.0)
            );
            assert_eq!(counts.get("returned_count").unwrap().as_f64(), Some(1.0));
            assert_eq!(json.get("physical_read_bytes").unwrap(), &JsonValue::Null);
            assert_eq!(json.get("staging_read_bytes").unwrap().as_f64(), Some(0.0));
            if source == 0 {
                assert_eq!(trace.value(Stage::Recall, Counter::IndexReadBytes), 0);
                if query == "needle" {
                    assert_eq!(
                        trace.value(Stage::ScoreCompletion, Counter::TextDocumentVisits),
                        1
                    );
                }
            } else if source == 1 {
                assert!(trace.value(Stage::Recall, Counter::IndexReadBytes) > 0);
                if query == "needle" {
                    assert_eq!(
                        trace.value(Stage::ScoreCompletion, Counter::TextDocumentVisits),
                        102
                    );
                }
            } else {
                assert_eq!(trace.total(Counter::IndexReadBytes), 0);
                assert!(trace.value(Stage::Recall, Counter::PrimaryReadBytes) > 0);
                if query == "needle" {
                    assert_eq!(
                        trace.value(Stage::ScoreCompletion, Counter::TextDocumentVisits),
                        52
                    );
                }
            }
            assert_eq!(
                trace.value(Stage::ScoreCompletion, Counter::PrimaryReadBytes),
                trace.value(Stage::ScoreCompletion, Counter::PrimaryRequestedBytes)
            );
            let snapshot = trace.total(Counter::VectorChecks);
            context
                .with_search_cost(&trace)
                .search_cost_add(Counter::VectorChecks, 99);
            assert_eq!(trace.total(Counter::VectorChecks), snapshot); // Search scope stopped all retained handles.
            let ordinary = results.to_json();
            let (control, control_work) = db
                .search_in_collection(
                    &id,
                    Some(query),
                    Some(&embedding),
                    &JsonValue::object(),
                    weights,
                    false,
                    None,
                    1,
                    Some(ResourceProfile::ResourceFirst),
                    Some(context.clone()),
                )
                .unwrap();
            assert_eq!(control.to_json(), ordinary); // Optional measurement does not change observable ranking/schema.
            assert_eq!(trace.total(Counter::VectorChecks), snapshot);
            drop((control, control_work));
            assert!(ordinary.get("search_cost").is_none());
            assert!(ordinary
                .get("query_context")
                .unwrap()
                .get("search_cost")
                .is_none());
            drop((results, retained));
            assert_eq!(context.available_workspace_bytes(), available);
        }
        drop(db);
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn failed_search_scope_disables_trace_without_inventing_candidate_counts() {
    let path = temp_path("query-cost-early-failure");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let trace = SearchCostTrace::new();
    let context = db.json_context().with_search_cost(&trace);
    assert_eq!(
        db.search_in_collection_traced(
            "missing",
            Some("needle"),
            None,
            &JsonValue::object(),
            None,
            false,
            None,
            0,
            None,
            Some(context.clone()),
            Some(&trace)
        )
        .err()
        .unwrap()
        .code(),
        ErrorCode::InvalidSearchParams
    );
    context.search_cost_add(Counter::VectorChecks, 1);
    context.search_cost_count(Count::Returned, 1);
    assert_eq!(trace.total(Counter::VectorChecks), 0);
    let json = trace.json(&JsonBuilder::host()).unwrap();
    assert!(json
        .get("candidate_counts")
        .unwrap()
        .get("returned_count")
        .unwrap()
        .is_null());
    drop(db);
    fs::remove_file(path).unwrap();
}
