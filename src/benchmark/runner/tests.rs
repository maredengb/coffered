use super::*;
use crate::cli::run;
use crate::json::{object_from_pairs, parse};
use crate::model::{
    ChunkInput, CollectionConfig, DistanceMetric, DocumentInput, EmbeddingInput,
    QueryEmbeddingInput, VectorIndexConfig,
};
use std::collections::BTreeSet;

fn insert_object_field(value: &mut JsonValue, key: &str, item: JsonValue) {
    value
        .as_object_mut()
        .expect("CLI command result is always a JSON object")
        .insert(key.into(), item);
}

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("cofferdb-cli-{name}-{}.coffer", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

fn detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
    err.detail()
        .iter()
        .find(|(field, _)| field == key)
        .map(|(_, value)| value.as_str())
}

#[test]
fn sqlite_match_query_streaming_preserves_terms_quotes_and_empty_queries() {
    let cases: &[(&str, &[&str])] = &[
        ("", &[]),
        ("THE and with", &[]),
        ("\"\".../", &[]),
        ("rust rust", &["rust", "rust"]),
        ("foo\"bar", &["foo", "bar"]),
        ("src/my_file.rs", &["src", "my_file", "my", "file", "rs"]),
        ("fooFoo", &["foofoo", "foo"]),
        ("HTTPServer", &["httpserver", "http", "server"]),
        (
            "\u{39f}\u{3a3}Value",
            &["\u{3bf}\u{3c3}value", "\u{3bf}\u{3c2}", "value"],
        ),
        ("\u{130}", &["i\u{307}"]),
        ("\u{4e2d}", &["\u{4e2d}"]),
    ];
    for (input, terms) in cases {
        let expected = (!terms.is_empty()).then(|| {
            terms
                .iter()
                .map(|term| format!("\"{term}\""))
                .collect::<Vec<_>>()
                .join(" OR ")
        });
        let context = JsonOperationContext::standalone();
        let actual = sqlite_fts5_match_query_in(input, &context).unwrap();
        assert_eq!(
            actual.as_ref().map(AsRef::as_ref),
            expected.as_deref(),
            "{input}"
        );
    }
    for seed in 0..100 {
        let mut input = String::new();
        let mut terms = Vec::new();
        for position in 0..=seed {
            let (part, expected) = cases[(position * 7 + seed) % cases.len()];
            input.push_str(part);
            input.push(' ');
            terms.extend_from_slice(expected);
        }
        let expected = (!terms.is_empty()).then(|| {
            terms
                .iter()
                .map(|term| format!("\"{term}\""))
                .collect::<Vec<_>>()
                .join(" OR ")
        });
        let context = JsonOperationContext::standalone();
        let actual = sqlite_fts5_match_query_in(&input, &context).unwrap();
        assert_eq!(
            actual.as_ref().map(AsRef::as_ref),
            expected.as_deref(),
            "seed {seed}"
        );
    }
}

#[test]
fn generated_import_manifest_uses_retained_benchmark_budget_and_optional_fields() {
    let (mut file, source) = create_temporary_file("manifest-budget-source", "jsonl").unwrap();
    file.write_all(b"{}\n").unwrap();
    file.flush().unwrap();
    let hash = crate::hash::sha256_hex(b"{}\n");
    let mut dataset = BenchmarkDataset {
        manifest: crate::benchmark::BenchmarkDatasetManifest {
            tokenizer: Default::default(),
            dataset_id: "generated".into(),
            version: "0.1".into(),
            chunking: JsonValue::Null,
            embedding_model: Some("model".into()),
            embedding_dim: 2,
            documents_sha256: hash.clone(),
            chunks_sha256: hash.clone(),
            embeddings_sha256: Some(hash.clone()),
            queries_sha256: hash.clone(),
            qrels_sha256: hash.clone(),
            query_embedding_source: QueryEmbeddingSource::Inline,
            query_embedding_model: Some("model".into()),
            query_embedding_dim: Some(2),
            query_embeddings_sha256: None,
            dataset_manifest_sha256: hash.clone(),
        },
        documents_path: source.path().to_path_buf(),
        chunks_path: source.path().to_path_buf(),
        embeddings_path: None,
        queries: Vec::new(),
        base_dir: PathBuf::new(),
        manifest_path_label: "dataset.json".into(),
    };
    let context = JsonOperationContext::standalone();
    let mut busy = context.reservation();
    busy.grow(
        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
        JsonInputKind::BenchmarkArtifact,
    )
    .unwrap();
    let error = write_benchmark_import_manifest(&dataset, "notes", &context).unwrap_err();
    assert_eq!(detail_value(&error, "json_input_kind"), Some("manifest"));
    assert_eq!(
        detail_value(&error, "json_limit_reason"),
        Some("work_budget_exceeded")
    );
    busy.release(32 * 1024);
    for seed in 0..100 {
        dataset.manifest.dataset_id = format!("dataset-{seed}");
        dataset.embeddings_path = (seed % 2 == 0).then(|| source.path().to_path_buf());
        let staged = write_benchmark_import_manifest(&dataset, "notes", &context).unwrap();
        let generated = fs::read_to_string(staged.path()).unwrap();
        let value = parse(&generated).unwrap();
        assert_eq!(
            value.get("dataset_id").and_then(JsonValue::as_str),
            Some(dataset.manifest.dataset_id.as_str())
        );
        assert_eq!(
            value.get("collection").and_then(JsonValue::as_str),
            Some("notes")
        );
        assert_eq!(
            value.get("tokenizer").unwrap(),
            &dataset.manifest.tokenizer.to_json()
        );
        let hashes = value.get("hashes").unwrap();
        assert_eq!(
            hashes.get("documents_sha256").and_then(JsonValue::as_str),
            Some(hash.as_str())
        );
        assert_eq!(value.get("embeddings").is_some(), seed % 2 == 0);
        assert_eq!(hashes.get("embeddings_sha256").is_some(), seed % 2 == 0);
        let staged_path = staged.path().to_path_buf();
        drop(staged);
        assert!(!staged_path.exists());
    }
    busy.grow(32 * 1024, JsonInputKind::BenchmarkArtifact)
        .unwrap();
}

#[test]
fn sqlite_match_query_exact_combined_peak_is_charged_on_both_axes() {
    use crate::json::WorkLedger;
    let payload = "PROJECT".len();
    let output = payload + 2;
    let minimum = payload + output;
    for parent in [false, true] {
        for limit in [minimum - 1, minimum, minimum + 1] {
            let ledger = std::sync::Arc::new(WorkLedger::with_limits(
                if parent { 2 * minimum } else { limit },
                0,
                if parent { limit } else { 2 * minimum },
            ));
            let context = JsonOperationContext::new(ledger.clone());
            let (result, keys) = crate::test_alloc::key_allocations(payload, || {
                sqlite_fts5_match_query_in("PROJECT", &context)
            });
            if limit < minimum {
                let error = result.err().unwrap();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "memory".into()),
                    ("operation", "operation_scratch".into()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", minimum.to_string()),
                ] {
                    assert!(error.detail().contains(&(key.into(), value)), "{error}");
                }
                // The sizing pass is admitted; the refused render pass never
                // allocates its normalized payload, and its output is discarded.
                assert_eq!(keys.allocated, 1);
                assert_eq!(context.available_workspace_bytes(), limit);
            } else {
                let query = result.unwrap().unwrap();
                assert_eq!(query.as_ref(), "\"project\"");
                assert_eq!(keys.allocated, 2);
                assert_eq!(context.available_workspace_bytes(), limit - output);
                let other = JsonOperationContext::new(ledger);
                drop(context);
                assert_eq!(other.available_workspace_bytes(), limit - output);
                drop(query);
                assert_eq!(other.available_workspace_bytes(), limit);
            }
        }
    }
    let context = JsonOperationContext::new(std::sync::Arc::new(WorkLedger::with_limits(0, 0, 0)));
    for text in ["", "THE and with", "\"\".../"] {
        assert!(sqlite_fts5_match_query_in(text, &context)
            .unwrap()
            .is_none());
    }
}

#[test]
fn sqlite_match_query_render_failures_discard_an_already_written_prefix() {
    use crate::json::{ReservedString, WorkLedger};
    let output = "\"rust\" OR \"project\"".len();
    let payload = "PROJECT".len();
    let minimum = output + payload;
    for parent in [false, true] {
        let limit = minimum - 1;
        let context = JsonOperationContext::new(std::sync::Arc::new(WorkLedger::with_limits(
            if parent { 2 * minimum } else { limit },
            0,
            if parent { limit } else { 2 * minimum },
        )));
        let (result, keys) = crate::test_alloc::key_allocations(payload, || {
            let mut query = ReservedString::with_capacity(output, &context, "operation_scratch")?;
            render_sqlite_fts5_match_query_in("rust PROJECT", &mut query, &context)?;
            Ok::<_, Error>(query)
        });
        let error = result.err().unwrap();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(
            detail_value(&error, "limit_bytes"),
            Some(limit.to_string().as_str())
        );
        assert_eq!(
            detail_value(&error, "requested_bytes"),
            Some(minimum.to_string().as_str())
        );
        assert_eq!(keys.allocated, 0);
        assert_eq!(context.available_workspace_bytes(), limit);
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let result = crate::test_alloc::fail_byte_allocation_once(payload, || {
            let mut query = ReservedString::with_capacity(output, &context, "operation_scratch")?;
            render_sqlite_fts5_match_query_in("rust PROJECT", &mut query, &context)?;
            Ok::<_, Error>(query)
        });
        assert_eq!(result.err().unwrap().code(), ErrorCode::IoError);
        assert_eq!(context.available_workspace_bytes(), capacity);
        assert_eq!(
            sqlite_fts5_match_query_in("rust PROJECT", &context)
                .unwrap()
                .unwrap()
                .as_ref(),
            "\"rust\" OR \"project\""
        );
    }
}

#[test]
fn benchmark_borrowed_queries_match_public_search_and_share_the_input_budget() {
    let path = temp_path("benchmark-borrowed-inputs");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
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
            metadata: parse(r#"{"tenant":"test"}"#).unwrap(),
            chunks: (0..4)
                .map(|index| ChunkInput {
                    chunk_external_id: format!("chunk-{index}"),
                    ordinal: index,
                    text: "wal rust ".repeat(index + 1),
                    metadata: object_from_pairs([("rank", JsonValue::Number(index as f64))]),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
    collection
        .put_embeddings(
            (0..4)
                .map(|index| EmbeddingInput {
                    chunk_external_id: format!("chunk-{index}"),
                    model_name: "m".into(),
                    dim: 2,
                    vector: vec![index as f32 + 1.0, 1.0],
                })
                .collect(),
        )
        .unwrap();
    let dataset = BenchmarkDataset {
        manifest: crate::benchmark::BenchmarkDatasetManifest {
            tokenizer: Default::default(),
            dataset_id: "borrowed".into(),
            version: "0.1".into(),
            chunking: JsonValue::Null,
            embedding_model: Some("m".into()),
            embedding_dim: 2,
            documents_sha256: String::new(),
            chunks_sha256: String::new(),
            embeddings_sha256: None,
            queries_sha256: String::new(),
            qrels_sha256: String::new(),
            query_embedding_source: QueryEmbeddingSource::Inline,
            query_embedding_model: Some("m".into()),
            query_embedding_dim: Some(2),
            query_embeddings_sha256: None,
            dataset_manifest_sha256: String::new(),
        },
        documents_path: PathBuf::new(),
        chunks_path: PathBuf::new(),
        embeddings_path: None,
        base_dir: PathBuf::new(),
        manifest_path_label: "dataset.json".into(),
        queries: (0..100)
            .map(|index| BenchmarkQuery {
                query_id: format!("q-{index}"),
                text: Some(if index % 2 == 0 { "wal" } else { "rust" }.into()),
                filter: parse(&format!(
                    "{{\"doc.tenant\":\"test\",\"chunk.rank\":{{\"$gte\":{}}}}}",
                    index % 5
                ))
                .unwrap(),
                top_k: if index % 5 == 0 { 600 } else { index % 4 + 1 },
                embedding: Some(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![(index % 3) as f32 + 1.0, 1.0],
                }),
                relevant_chunk_ids: BTreeSet::from(["chunk-3".into()]),
            })
            .collect(),
    };
    let capacity = db.json_context().available_workspace_bytes();
    let context = JsonOperationContext::standalone();
    let mut input_work = context.reservation();
    for query in &dataset.queries {
        input_work.merge(
            query
                .filter
                .guard_in(JsonInputKind::Filter, &context)
                .unwrap()
                .0,
        );
        input_work.merge(
            query
                .embedding
                .as_ref()
                .unwrap()
                .reserve_json_work_in(&context)
                .unwrap(),
        );
    }
    let before = fs::read(&path).unwrap();
    for profile in [
        ResourceProfile::Expanded,
        ResourceProfile::Balanced,
        ResourceProfile::Constrained,
    ] {
        for mode in [
            BenchmarkQueryMode::TextBm25,
            BenchmarkQueryMode::VectorExact,
            BenchmarkQueryMode::Hybrid,
        ] {
            let (run, metrics, run_work) = run_benchmark_queries(
                &db, "notes", &dataset, profile, mode, "warm", &context, None,
            )
            .unwrap();
            let expected_work = run
                .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &context)
                .unwrap();
            assert_eq!(run_work.reserved_bytes(), expected_work.reserved_bytes());
            drop(expected_work);
            let rows = run.get("per_query").unwrap().as_array().unwrap();
            let mut expected_scans = 0;
            let mut expected_distances = 0;
            for (row, query) in rows.iter().zip(&dataset.queries) {
                let mut builder = collection
                    .search()
                    .top_k(query.top_k)
                    .filter(query.filter.clone())
                    .resource_profile(profile);
                if mode.has_text() {
                    builder = builder.text(query.text.as_ref().unwrap().clone());
                }
                if mode.has_vector() {
                    builder = builder.query_embedding(query.embedding.as_ref().unwrap().clone());
                }
                let expected = builder.execute().unwrap();
                let limit = query.top_k.max(profile.params().bm25_candidates);
                assert_eq!(expected.query_context.candidate_limits.bm25, limit);
                assert_eq!(expected.query_context.candidate_limits.vector, limit);
                assert_eq!(expected.query_context.candidate_limits.fusion, limit);
                assert_eq!(
                    row.get("ranked_chunk_ids").unwrap(),
                    &JsonValue::Array(
                        expected
                            .results
                            .iter()
                            .map(|hit| JsonValue::String(hit.chunk_external_id.clone()))
                            .collect()
                    )
                );
                let actual_scores = row.get("scores").unwrap().as_array().unwrap();
                for (actual, expected) in actual_scores.iter().zip(&expected.results) {
                    assert_eq!(
                        actual.as_f64().unwrap().to_bits(),
                        expected.final_score.to_bits()
                    );
                }
                assert_eq!(
                    row.get("query_context").unwrap(),
                    &expected.query_context.to_json()
                );
                let stats = benchmark_execution_stats(&expected);
                assert_eq!(row.get("execution_stats").unwrap(), &stats.to_json());
                expected_scans += stats.vector_scan_count;
                expected_distances += stats.distance_ops;
            }
            assert_eq!(rows.len(), 100);
            assert_eq!(metrics.vector_scan_count, expected_scans);
            assert_eq!(metrics.distance_ops, expected_distances);
        }
    }
    let config = db.list_collections().unwrap().remove(0);
    let config_bytes = config.collection_id.len()
        + config.name.len()
        + config.embedding_model.as_ref().map_or(0, String::len)
        + config.created_at.len();
    let mut busy = context.reservation();
    busy.grow(
        capacity - input_work.reserved_bytes(),
        JsonInputKind::Filter,
    )
    .unwrap();
    let error = run_benchmark_queries(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        BenchmarkQueryMode::Hybrid,
        "warm",
        &context,
        None,
    )
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    assert_eq!(detail_value(&error, "resource"), Some("memory"));
    assert_eq!(detail_value(&error, "operation"), Some("operation_scratch"));
    // The handle is handed off first; the query then holds its full configuration.
    busy.retain(busy.reserved_bytes() - config_bytes);
    let error = run_benchmark_queries(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        BenchmarkQueryMode::Hybrid,
        "warm",
        &context,
        None,
    )
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
    assert_eq!(
        detail_value(&error, "json_limit_reason"),
        Some("work_budget_exceeded")
    );
    assert_eq!(detail_value(&error, "json_input_kind"), Some("filter"));
    drop(busy);
    run_benchmark_queries(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        BenchmarkQueryMode::Hybrid,
        "warm",
        &context,
        None,
    )
    .unwrap();
    assert_eq!(fs::read(&path).unwrap(), before);
    drop(dataset);
    drop(input_work);
    context
        .reservation()
        .grow(capacity, JsonInputKind::Filter)
        .unwrap();
    drop(collection);
    drop(db);
    context
        .reservation()
        .grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::Filter,
        )
        .unwrap();
    fs::remove_file(path).unwrap();
}

pub(super) fn text_only_benchmark_dataset(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("documents.jsonl"),
        r#"{"external_id":"doc","metadata":{}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("chunks.jsonl"),
        concat!(
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"benchmark query","metadata":{"lang":"rust"}}"#,
            "\n",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#2","ordinal":2,"text":"benchmark query","metadata":{"lang":"python"}}"#
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("queries.jsonl"),
        r#"{"query_id":"q1","text":"benchmark query","filter":{"chunk.lang":"rust"},"top_k":10}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("qrels.jsonl"),
        r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
    )
    .unwrap();
    let documents_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("documents.jsonl")).unwrap());
    let chunks_sha256 = crate::hash::sha256_hex(&std::fs::read(dir.join("chunks.jsonl")).unwrap());
    let queries_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("queries.jsonl")).unwrap());
    let qrels_sha256 = crate::hash::sha256_hex(&std::fs::read(dir.join("qrels.jsonl")).unwrap());
    std::fs::write(
        dir.join("dataset_manifest.json"),
        format!(
            concat!(
                r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                r#""query_embedding_source":"none","hashes":{{"#,
                r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                r#"}}"#
            ),
            documents_sha256, chunks_sha256, queries_sha256, qrels_sha256
        ),
    )
    .unwrap();

    dir
}

#[test]
fn typed_runner_preserves_options_output_and_reservation_ownership() {
    let dir = text_only_benchmark_dataset("typed-runner-options");
    for profile in [
        ResourceProfile::Expanded,
        ResourceProfile::Balanced,
        ResourceProfile::Constrained,
    ] {
        for durability in [Durability::Commit, Durability::Fast] {
            let label = format!("{}-{}", profile.as_json_str(), durability.as_str());
            let database = dir.join(format!("{label}.coffer"));
            let result_file = dir.join(format!("{label}.json"));
            let output = run_benchmark(BenchmarkOptions {
                path: database.to_string_lossy().into_owned(),
                dataset_path: dir.to_string_lossy().into_owned(),
                collection: label.clone(),
                profile,
                durability,
                result_file: Some(result_file.to_string_lossy().into_owned()),
                require_baseline: false,
            })
            .unwrap();
            assert_eq!(
                output.value.get("profile").and_then(JsonValue::as_str),
                Some(profile.as_json_str())
            );
            assert_eq!(
                output.value.get("durability").and_then(JsonValue::as_str),
                Some(durability.as_str())
            );
            assert_eq!(
                output.value.get("collection").and_then(JsonValue::as_str),
                Some(label.as_str())
            );
            assert_eq!(
                parse(&fs::read_to_string(&result_file).unwrap()).unwrap(),
                output.value
            );
            validate_benchmark_result_contract(&output.value).unwrap();
            let runs = output.value.get("runs").unwrap().as_array().unwrap();
            assert_eq!(runs.len(), 2);
            for (run, state) in runs.iter().zip(["cold", "warm"]) {
                assert_eq!(run.get("cache_state").unwrap().as_str(), Some(state));
                let query = &run.get("per_query").unwrap().as_array().unwrap()[0];
                assert_eq!(
                    query.get("ranked_chunk_ids").unwrap(),
                    &JsonValue::Array(vec![JsonValue::String("doc#1".into())])
                );
                assert_eq!(
                    query
                        .get("query_context")
                        .unwrap()
                        .get("profile")
                        .unwrap()
                        .as_str(),
                    Some(profile.as_json_str())
                );
            }
            let expected = output
                .value
                .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &output.context)
                .unwrap();
            assert_eq!(output.work.reserved_bytes(), expected.reserved_bytes());
            drop(expected);
            let context = output.context.clone();
            drop(output);
            let mut work = context.reservation();
            work.grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
            let db = Database::open(&database).unwrap();
            assert_eq!(
                db.collection(&label).unwrap().stats().unwrap().chunk_count,
                2
            );
            db.close().unwrap();
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn typed_runner_checks_output_aliases_before_creating_database() {
    let dir = text_only_benchmark_dataset("typed-runner-aliases");
    let database = dir.join("never-created.coffer");
    for source in [
        "documents.jsonl",
        "chunks.jsonl",
        "queries.jsonl",
        "qrels.jsonl",
        "dataset_manifest.json",
    ] {
        let input = dir.join(source);
        let before = fs::read(&input).unwrap();
        let error = run_benchmark(BenchmarkOptions {
            path: database.to_string_lossy().into_owned(),
            dataset_path: dir.to_string_lossy().into_owned(),
            collection: "notes".into(),
            profile: ResourceProfile::Balanced,
            durability: Durability::Commit,
            result_file: Some(input.to_string_lossy().into_owned()),
            require_baseline: false,
        })
        .err()
        .expect("input overwrite must fail");
        assert_eq!(error.code(), ErrorCode::PermissionDenied);
        assert_eq!(fs::read(&input).unwrap(), before);
        assert!(!database.exists());
        assert!(!crate::lock::lock_path(&database).exists());
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn import_timing_keeps_the_original_clock_through_all_verified_queries() {
    let dir = text_only_benchmark_dataset("continuous-import-clock");
    let output = run_benchmark_started(
        BenchmarkOptions {
            path: dir.join("data.coffer").to_string_lossy().into_owned(),
            dataset_path: dir.to_string_lossy().into_owned(),
            collection: "notes".into(),
            profile: ResourceProfile::Constrained,
            durability: Durability::Fast,
            result_file: None,
            require_baseline: false,
        },
        Instant::now() - std::time::Duration::from_secs(10),
    )
    .unwrap();
    let elapsed = output
        .value
        .get("metrics")
        .unwrap()
        .get("import_time_ms")
        .unwrap();
    assert_eq!(
        Some(elapsed),
        output
            .value
            .get("import_context")
            .unwrap()
            .get("import_time_ms")
    );
    let query_time = output
        .value
        .get("runs")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|run| run.get("per_query").unwrap().as_array().unwrap())
        .map(|query| query.get("latency_ms").unwrap().as_f64().unwrap())
        .sum::<f64>();
    assert!(elapsed.as_f64().unwrap() >= 10_000.0 + query_time);
    fs::remove_dir_all(dir).unwrap();
}

fn read_execution_records(path: &Path) -> Vec<JsonValue> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| parse(line).unwrap())
        .collect()
}

#[test]
fn execution_completed_benchmark_publishes_linked_protocol_expectations_and_attempt_records() {
    let dir = text_only_benchmark_dataset("benchmark-execution-completed");
    let result_path = dir.join("result.json");
    let output = run_benchmark(BenchmarkOptions {
        path: dir.join("data.coffer").to_string_lossy().into_owned(),
        dataset_path: dir.to_string_lossy().into_owned(),
        collection: "notes".into(),
        profile: ResourceProfile::Constrained,
        durability: Durability::Fast,
        result_file: Some(result_path.to_string_lossy().into_owned()),
        require_baseline: false,
    })
    .unwrap();
    let paths = fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("cofferdb-benchmark-execution-")
        })
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 1);
    let records = read_execution_records(&paths[0]);
    assert_eq!(records[0].get("event").unwrap().as_str(), Some("protocol"));
    assert_eq!(
        records[0].get("rules_version").unwrap().as_str(),
        Some("cofferdb-builtin-source-acceptance-v4")
    );
    assert_eq!(
        records[0].get("result_file").unwrap().as_str(),
        result_path.to_str()
    );
    let events = records
        .iter()
        .map(|record| record.get("event").unwrap().as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        events,
        [
            "protocol",
            "planned",
            "planned",
            "import_completed",
            "expected",
            "started",
            "search_completed",
            "completed",
            "group_summary",
            "started",
            "search_completed",
            "completed",
            "group_summary",
            "run_completed"
        ]
    );
    assert_eq!(
        records[4].get("ranked_chunk_ids").unwrap(),
        &JsonValue::Array(vec![JsonValue::String("doc#1".into())])
    );
    let final_record = records.last().unwrap();
    assert_eq!(
        final_record
            .get("result_pretty_json_sha256")
            .unwrap()
            .as_str(),
        Some(crate::hash::sha256_hex(&fs::read(&result_path).unwrap()).as_str())
    );
    assert_eq!(
        final_record.get("import_time_ms"),
        output.value.get("metrics").unwrap().get("import_time_ms")
    );
    assert_eq!(
        final_record.get("completed_samples").unwrap().as_f64(),
        Some(2.0)
    );
    let phases = &records[3];
    let scan = crate::storage::scan_file(dir.join("data.coffer")).unwrap();
    assert_eq!(
        phases.get("committed_batches").unwrap().as_f64(),
        Some(scan.commits as f64)
    );
    assert_eq!(
        phases.get("checkpoint_cycles").unwrap().as_f64(),
        Some(scan.commits as f64)
    );
    assert_eq!(
        phases.get("checkpoint_ready_cycles").unwrap().as_f64(),
        Some(scan.commits as f64)
    );
    assert_eq!(
        phases.get("final_checkpoint_ready").unwrap().as_bool(),
        Some(true)
    );
    assert_eq!(phases.get("max_batch_chunks").unwrap().as_f64(), Some(2.0));
    let phase_time = phases.get("commit_time_ms").unwrap().as_f64().unwrap()
        + phases.get("checkpoint_time_ms").unwrap().as_f64().unwrap();
    assert!(phase_time >= 0.0 && phase_time <= phases.get("elapsed_ms").unwrap().as_f64().unwrap());
    assert!(
        phases.get("elapsed_ms").unwrap().as_f64().unwrap()
            <= final_record
                .get("import_time_ms")
                .unwrap()
                .as_f64()
                .unwrap()
    );
    for (record, run) in [7, 11]
        .into_iter()
        .map(|index| &records[index])
        .zip(output.value.get("runs").unwrap().as_array().unwrap())
    {
        let row = &run.get("per_query").unwrap().as_array().unwrap()[0];
        assert_eq!(record.get("elapsed_ms"), row.get("latency_ms"));
        let cost = record.get("search_cost").unwrap();
        assert_eq!(
            cost.get("version").unwrap().as_str(),
            Some(crate::search_cost::VERSION)
        );
        assert_eq!(
            cost.get("counter_saturated").unwrap().as_bool(),
            Some(false)
        );
        assert!(cost.get("physical_read_bytes").unwrap().is_null());
        assert_eq!(
            cost.get("candidate_counts")
                .unwrap()
                .get("returned_count")
                .unwrap()
                .as_f64(),
            Some(1.0)
        );
        assert_eq!(cost.get("stages").unwrap().as_array().unwrap().len(), 5);
        assert!(
            cost.get("totals")
                .unwrap()
                .get("primary_read_bytes")
                .unwrap()
                .as_f64()
                .unwrap()
                > 0.0
        );
        assert!(row
            .get("query_context")
            .unwrap()
            .get("search_cost")
            .is_none());
        assert_eq!(
            record.get("index_source"),
            row.get("query_context").unwrap().get("index_source")
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn imported_phase_survives_an_actual_query_read_failure() {
    use std::io::{Read, Seek, SeekFrom, Write};

    let dir = text_only_benchmark_dataset("import-phase-query-failure");
    let context = JsonOperationContext::standalone();
    let (dataset, _input_work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
    let database = dir.join("data.coffer");
    let result_path = dir.join("result.json");
    fs::write(&result_path, b"previous result").unwrap();
    let started = Instant::now();
    let db = Database::create(&database, DatabaseConfig::default()).unwrap();
    let manifest = write_benchmark_import_manifest(&dataset, "notes", &context).unwrap();
    let measured = db
        .import_manifest_measured_in(
            manifest.path(),
            OnConflict::Replace,
            crate::db::ImportIndexBuildMode::InlineFullCheckpoint,
            context.clone(),
        )
        .unwrap();
    let modes = [BenchmarkQueryMode::TextBm25];
    let acceptance = acceptance::Acceptance::prepare(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        &modes,
        &context,
    )
    .unwrap();
    let mut journal = execution::ExecutionJournal::new(
        &dataset,
        &modes,
        execution::ExecutionConfig {
            database: database.to_str().unwrap(),
            result_file: result_path.to_str(),
            collection: "notes",
            profile: ResourceProfile::Balanced,
            durability: Durability::Commit,
        },
        &context,
        started,
    )
    .unwrap();
    journal.import_completed(&measured.batches).unwrap();
    journal.expectations(&acceptance).unwrap();
    let scan = crate::storage::scan_file(&database).unwrap();
    let record = scan
        .records
        .iter()
        .find(|record| {
            matches!(&record.payload,
                crate::storage::RecordPayload::ChunkPut(chunk) if chunk.chunk_external_id == "doc#1"
            )
        })
        .unwrap();
    // Damage after reference preparation so the real query, not input validation, fails.
    let bytes = fs::read(&database).unwrap();
    let payload = &bytes[record.record_ref.payload.offset as usize
        ..(record.record_ref.payload.offset + record.record_ref.payload.len) as usize];
    let text_offset = payload
        .windows(b"benchmark query".len())
        .position(|window| window == b"benchmark query")
        .unwrap();
    let offset = record.record_ref.payload.offset + text_offset as u64;
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&database)
        .unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut byte = [0];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 1;
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_data().unwrap();
    drop(file);
    let error = run_benchmark_queries(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        BenchmarkQueryMode::TextBm25,
        "cold",
        &context,
        Some(QueryVerification {
            expected: &acceptance,
            mode: 0,
            journal: Some(&mut journal),
        }),
    )
    .err()
    .expect("a damaged committed chunk must fail the real query");
    assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
    let error = journal.fail(error);
    assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
    let rows = read_execution_records(
        &dir.join(detail_value(&error, "benchmark_execution_record").unwrap()),
    );
    let count = |event| {
        rows.iter()
            .filter(|row| row.get("event").unwrap().as_str() == Some(event))
            .count()
    };
    assert_eq!(count("import_completed"), 1);
    assert_eq!(count("started"), 1);
    assert_eq!(count("search_failed"), 1);
    let search = rows
        .iter()
        .find(|row| row.get("event").unwrap().as_str() == Some("search_failed"))
        .unwrap();
    let failed = rows
        .iter()
        .find(|row| row.get("event").unwrap().as_str() == Some("failed"))
        .unwrap();
    assert_eq!(search.get("elapsed_ms"), failed.get("elapsed_ms"));
    assert_eq!(search.get("search_cost"), failed.get("search_cost"));
    let cost = failed.get("search_cost").unwrap();
    assert!(
        cost.get("totals")
            .unwrap()
            .get("primary_read_bytes")
            .unwrap()
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(cost
        .get("candidate_counts")
        .unwrap()
        .get("returned_count")
        .unwrap()
        .is_null());
    assert_eq!(count("failed"), 1);
    assert_eq!(count("not_run"), 1);
    assert_eq!(count("completed"), 0);
    assert_eq!(count("run_completed"), 0);
    assert_eq!(
        rows.last().unwrap().get("event").unwrap().as_str(),
        Some("run_failed")
    );
    assert_eq!(fs::read(&result_path).unwrap(), b"previous result");
    drop((journal, acceptance, manifest, db));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn execution_failures_preserve_completed_and_unexecuted_samples_across_100_boundaries() {
    let dir = text_only_benchmark_dataset("benchmark-execution-failures");
    let context = JsonOperationContext::standalone();
    let (mut dataset, _input_work) =
        BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
    dataset.queries = (0..12)
        .map(|index| {
            let mut query = dataset.queries[0].clone();
            query.query_id = format!("query-{index}");
            query
        })
        .collect();
    let database = dir.join("data.coffer");
    let db = Database::create(&database, DatabaseConfig::default()).unwrap();
    let manifest = write_benchmark_import_manifest(&dataset, "notes", &context).unwrap();
    db.import_manifest(manifest.path(), OnConflict::Replace)
        .unwrap();
    let results = db
        .collection("notes")
        .unwrap()
        .search()
        .text("benchmark query")
        .execute()
        .unwrap();
    let modes = [BenchmarkQueryMode::TextBm25];
    let mut previous = BTreeSet::new();
    for seed in 0..100 {
        let completed = seed % 25;
        let during_query = seed % 2 == 0 && completed < 24;
        let mut journal = execution::ExecutionJournal::new(
            &dataset,
            &modes,
            execution::ExecutionConfig {
                database: database.to_str().unwrap(),
                result_file: None,
                collection: "notes",
                profile: ResourceProfile::Balanced,
                durability: Durability::Fast,
            },
            &context,
            Instant::now(),
        )
        .unwrap();
        for index in 0..completed {
            journal
                .begin(0, if index < 12 { "cold" } else { "warm" }, index % 12)
                .unwrap();
            assert!(journal
                .complete_query(1.0 + index as f64, &results)
                .is_err());
            // This boundary test supplies synthetic search measurements; the
            // real search/read-cost behavior is covered by the query/fault tests.
            let trace = crate::search_cost::SearchCostTrace::new();
            trace.finish();
            journal
                .search_finished(trace, 1.0 + index as f64, true)
                .unwrap();
            assert!(journal
                .complete_query(2.0 + index as f64, &results)
                .is_err());
            journal
                .complete_query(1.0 + index as f64, &results)
                .unwrap();
        }
        if during_query {
            journal
                .begin(
                    0,
                    if completed < 12 { "cold" } else { "warm" },
                    completed % 12,
                )
                .unwrap();
        }
        let error = journal.fail(
            Error::new(ErrorCode::ChecksumMismatch, "injected sample failure")
                .with_detail("original", "kept"),
        );
        assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        assert_eq!(detail_value(&error, "original"), Some("kept"));
        let name = detail_value(&error, "benchmark_execution_record").unwrap();
        assert_eq!(Path::new(name).components().count(), 1);
        let path = dir.join(name);
        assert!(
            previous.insert(path.clone()),
            "new runs must not overwrite prior attempts"
        );
        let records = read_execution_records(&path);
        let by_event = |event: &'static str| {
            records
                .iter()
                .filter(move |record| record.get("event").unwrap().as_str() == Some(event))
        };
        assert_eq!(by_event("planned").count(), 24);
        assert_eq!(by_event("completed").count(), completed);
        assert_eq!(by_event("failed").count(), usize::from(during_query));
        assert_eq!(
            by_event("not_run").count(),
            24 - completed - usize::from(during_query)
        );
        assert!(
            by_event("not_run").all(|record| record.get("elapsed_ms") == Some(&JsonValue::Null))
        );
        assert_eq!(
            records.last().unwrap().get("event").unwrap().as_str(),
            Some("run_failed")
        );
        assert_eq!(by_event("run_completed").count(), 0);
        let summaries = by_event("group_summary").collect::<Vec<_>>();
        assert_eq!(summaries.len(), 2);
        for (group, summary) in summaries.iter().enumerate() {
            let done = completed.saturating_sub(group * 12).min(12);
            assert_eq!(
                summary.get("completed").unwrap().as_f64(),
                Some(done as f64)
            );
            for (field, percentile) in [("p50_ms", 0.5_f64), ("p95_ms", 0.95_f64)] {
                if done == 0 {
                    assert_eq!(summary.get(field), Some(&JsonValue::Null));
                } else {
                    assert_eq!(
                        summary.get(field).unwrap().as_f64(),
                        Some((group * 12) as f64 + (percentile * done as f64).ceil())
                    );
                }
            }
            assert_eq!(
                summary.get("statistics_scope").unwrap().as_str(),
                Some(if done == 12 {
                    "complete_group"
                } else {
                    "diagnostic_successful_samples_only"
                })
            );
        }
    }
    drop((manifest, db));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn benchmark_generated_input_errors_keep_public_locators_without_private_paths() {
    let dir = text_only_benchmark_dataset("benchmark-private-input-error");
    let dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
    for seed in 0..100_u64 {
        let (original_path, expected_path) = match seed % 3 {
            0 => (dataset.documents_path.clone(), Some("documents.jsonl")),
            1 => (
                fs::canonicalize(&dataset.chunks_path).unwrap(),
                Some("chunks.jsonl"),
            ),
            _ => (
                PathBuf::from("/private/unpublished-import-manifest.json"),
                None,
            ),
        };
        let error = Error::resource_exhausted("memory", "batch_prepare", seed, u128::MAX)
            .with_detail("path", original_path.display())
            .with_u64_detail("line", seed + 1);
        let original = error.to_json();
        let mapped = dataset.locate_input_error(error);
        assert_eq!(detail_value(&mapped, "path"), expected_path);
        assert_eq!(detail_value(&mapped, "manifest_path"), dir.to_str());
        let json = mapped.to_json();
        for field in [
            "resource",
            "operation",
            "limit_bytes",
            "requested_bytes",
            "line",
        ] {
            assert_eq!(
                original.get("detail").unwrap().get(field),
                json.get("detail").unwrap().get(field)
            );
        }
        assert!(!json
            .to_compact_string()
            .contains("/private/unpublished-import-manifest.json"));
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn benchmark_valid_long_reference_artifacts_do_not_invent_catalog_quota() {
    let dir = text_only_benchmark_dataset("benchmark-catalog-resource-error");
    let id = "d".repeat(256 * 1024);
    let documents = format!(r#"{{"external_id":"{id}","metadata":{{}}}}"#);
    let chunks = format!(
        r#"{{"document_external_id":"{id}","chunk_external_id":"doc#1","ordinal":1,"text":"benchmark query","metadata":{{"lang":"rust"}}}}"#
    );
    let mut manifest =
        parse(&fs::read_to_string(dir.join("dataset_manifest.json")).unwrap()).unwrap();
    for (name, content) in [("documents", documents), ("chunks", chunks)] {
        fs::write(dir.join(format!("{name}.jsonl")), &content).unwrap();
        manifest
            .as_object_mut()
            .unwrap()
            .get_mut("hashes")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                format!("{name}_sha256"),
                JsonValue::String(crate::hash::sha256_hex(content.as_bytes())),
            );
    }
    fs::write(
        dir.join("dataset_manifest.json"),
        manifest.to_compact_string(),
    )
    .unwrap();
    BenchmarkDataset::load_and_validate(&dir).unwrap();
    let database = dir.join("data.coffer");
    let output = run_benchmark(BenchmarkOptions {
        path: database.to_string_lossy().into_owned(),
        dataset_path: dir.to_string_lossy().into_owned(),
        collection: "notes".into(),
        profile: ResourceProfile::Constrained,
        durability: Durability::Fast,
        result_file: None,
        require_baseline: false,
    })
    .unwrap();
    let paths = fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("cofferdb-benchmark-execution-")
        })
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 1);
    let records = read_execution_records(&paths[0]);
    assert_eq!(
        records.last().unwrap().get("event").unwrap().as_str(),
        Some("run_completed")
    );
    assert_eq!(
        records
            .iter()
            .filter(|row| row.get("event").unwrap().as_str() == Some("completed"))
            .count(),
        2
    );
    assert!(!records.iter().any(|row| matches!(
        row.get("event").unwrap().as_str(),
        Some("failed" | "not_run" | "run_failed")
    )));
    let db = Database::open_read_only(&database).unwrap();
    let collection = db.collection("notes").unwrap();
    assert_eq!(
        collection
            .get_document_by_external_id(&id)
            .unwrap()
            .external_id,
        id
    );
    assert_eq!(collection.stats().unwrap().chunk_count, 1);
    db.close().unwrap();
    drop(output);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn benchmark_real_catalog_budget_failure_preserves_unexecuted_plan() {
    let dir = text_only_benchmark_dataset("benchmark-real-catalog-resource-error");
    let database = dir.join("data.coffer");
    let db = Database::create(
        &database,
        DatabaseConfig {
            resource_profile: ResourceProfile::Constrained,
            durability: Durability::Fast,
            ..DatabaseConfig::default()
        },
    )
    .unwrap();
    let context = db.json_context();
    let (dataset, _dataset_work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
    let modes = benchmark_query_modes(&dataset);
    let mut journal = execution::ExecutionJournal::new(
        &dataset,
        &modes,
        execution::ExecutionConfig {
            database: database.to_str().unwrap(),
            result_file: None,
            collection: "notes",
            profile: ResourceProfile::Constrained,
            durability: Durability::Fast,
        },
        &context,
        Instant::now(),
    )
    .unwrap();
    let manifest = write_benchmark_import_manifest(&dataset, "notes", &context).unwrap();
    let mut busy = context.catalog_reservation();
    let limit = crate::json::WorkLedger::CATALOG_LIMIT_BYTES;
    busy.grow(context.available_catalog_bytes()).unwrap();
    let probe = busy.grow(1).unwrap_err();
    assert_eq!(detail_value(&probe, "resource"), Some("catalog"));
    assert_eq!(
        detail_value(&probe, "limit_bytes"),
        Some(limit.to_string().as_str())
    );
    assert_eq!(
        detail_value(&probe, "requested_bytes"),
        Some((limit + 1).to_string().as_str())
    );
    let before = fs::read(&database).unwrap();
    let error = db
        .import_manifest_measured_in(
            manifest.path(),
            OnConflict::Replace,
            crate::db::ImportIndexBuildMode::InlineFullCheckpoint,
            context.clone(),
        )
        .err()
        .unwrap();
    let error = journal.fail(dataset.locate_input_error(error));
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    assert_eq!(detail_value(&error, "resource"), Some("catalog"));
    assert_eq!(detail_value(&error, "operation"), Some("catalog_build"));
    assert_eq!(
        detail_value(&error, "limit_bytes"),
        Some(limit.to_string().as_str())
    );
    assert!(
        detail_value(&error, "requested_bytes")
            .unwrap()
            .parse::<u128>()
            .unwrap()
            > limit as u128
    );
    assert_eq!(detail_value(&error, "path"), None);
    assert_eq!(detail_value(&error, "manifest_path"), dir.to_str());
    assert!(!crate::json::is_limit_error(&error));
    assert!(!error.to_string().contains("generated-import-manifest"));
    let records = read_execution_records(
        &dir.join(detail_value(&error, "benchmark_execution_record").unwrap()),
    );
    assert_eq!(
        records
            .iter()
            .filter(|row| row.get("event").unwrap().as_str() == Some("not_run"))
            .count(),
        2
    );
    assert_eq!(
        records.last().unwrap().get("error_code").unwrap().as_str(),
        Some("resource_exhausted")
    );
    assert!(!records
        .iter()
        .any(|row| row.get("event").unwrap().as_str() == Some("import_completed")));
    assert_eq!(fs::read(&database).unwrap(), before);
    assert!(db.list_collections().unwrap().is_empty());
    assert_eq!(db.temporary_disk_budget().used(), 0);
    drop(journal);
    drop((busy, manifest, dataset, _dataset_work));
    db.close().unwrap();
    assert_eq!(
        context.available_total_bytes(),
        JsonOperationContext::standalone().available_total_bytes()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn execution_failure_keeps_prior_result_and_reports_imported_progress_without_success() {
    let dir = text_only_benchmark_dataset("benchmark-execution-source-mismatch");
    let database = dir.join("data.coffer");
    let db = Database::create(&database, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 1,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "outside-fixed-dataset".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: Vec::new(),
        })
        .unwrap();
    drop((collection, db));
    let result_path = dir.join("result.json");
    fs::write(&result_path, b"previous result").unwrap();
    let error = run_benchmark(BenchmarkOptions {
        path: database.to_string_lossy().into_owned(),
        dataset_path: dir.to_string_lossy().into_owned(),
        collection: "notes".into(),
        profile: ResourceProfile::Balanced,
        durability: Durability::Fast,
        result_file: Some(result_path.to_string_lossy().into_owned()),
        require_baseline: false,
    })
    .err()
    .unwrap();
    assert_eq!(
        detail_value(&error, "reason"),
        Some("benchmark_acceptance_mismatch")
    );
    assert!(
        detail_value(&error, "benchmark_elapsed_ms")
            .unwrap()
            .parse::<f64>()
            .unwrap()
            > 0.0
    );
    assert_eq!(fs::read(&result_path).unwrap(), b"previous result");
    let records = read_execution_records(
        &dir.join(detail_value(&error, "benchmark_execution_record").unwrap()),
    );
    assert_eq!(
        records
            .iter()
            .filter(|row| row.get("event").unwrap().as_str() == Some("not_run"))
            .count(),
        2
    );
    assert_eq!(
        records.last().unwrap().get("event").unwrap().as_str(),
        Some("run_failed")
    );
    assert!(records
        .iter()
        .any(|row| row.get("event").unwrap().as_str() == Some("import_completed")));
    let db = Database::open(&database).unwrap();
    assert_eq!(
        db.collection("notes")
            .unwrap()
            .stats()
            .unwrap()
            .document_count,
        2
    );
    db.close().unwrap();
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_acceptance_completes_cross_window_scores_before_fusion_truncation() {
    for branch in ["text", "vector"] {
        let dir = text_only_benchmark_dataset(&format!("source-complete-window-{branch}"));
        let mut chunks = String::new();
        let mut embeddings = String::new();
        for i in 0..51 {
            let (text, vector) = match (branch, i == 50) {
                ("text", true) => ("needle filler", "[1,0]"),
                ("text", false) => ("needle", "[-1,0]"),
                ("vector", true) => ("needle needle needle needle", "[0,1]"),
                ("vector", false) => ("needle filler filler filler", "[4,3]"),
                _ => unreachable!(),
            };
            chunks.push_str(&format!(
            "{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"doc#{i}\",\"ordinal\":{i},\"text\":\"{text}\",\"metadata\":{{}}}}\n",
        ));
            embeddings.push_str(&format!(
            "{{\"chunk_external_id\":\"doc#{i}\",\"model_name\":\"m\",\"dim\":2,\"vector\":{vector}}}\n",
        ));
        }
        fs::write(dir.join("chunks.jsonl"), &chunks).unwrap();
        fs::write(dir.join("embeddings.jsonl"), &embeddings).unwrap();
        let mut manifest =
            parse(&fs::read_to_string(dir.join("dataset_manifest.json")).unwrap()).unwrap();
        manifest
            .as_object_mut()
            .unwrap()
            .get_mut("hashes")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "chunks_sha256".into(),
                JsonValue::String(crate::hash::sha256_hex(chunks.as_bytes())),
            );
        fs::write(
            dir.join("dataset_manifest.json"),
            manifest.to_compact_string(),
        )
        .unwrap();
        let context = JsonOperationContext::standalone();
        let (mut dataset, input_work) =
            BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
        dataset.embeddings_path = Some(dir.join("embeddings.jsonl"));
        dataset.manifest.embeddings_sha256 = Some(crate::hash::sha256_hex(embeddings.as_bytes()));
        dataset.manifest.embedding_dim = 2;
        dataset.manifest.embedding_model = Some("m".into());
        dataset.queries = vec![BenchmarkQuery {
            query_id: "cross-window".into(),
            text: Some("needle".into()),
            filter: JsonValue::Null,
            embedding: Some(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            }),
            top_k: 1,
            relevant_chunk_ids: BTreeSet::new(),
        }];
        let modes = benchmark_query_modes(&dataset);
        let db = Database::create(dir.join("data.coffer"), DatabaseConfig::default()).unwrap();
        let import_manifest = write_benchmark_import_manifest(&dataset, "notes", &context).unwrap();
        db.import_manifest(import_manifest.path(), OnConflict::Error)
            .unwrap();
        drop(import_manifest);
        let profile = ResourceProfile::Constrained;
        let query = &dataset.queries[0];
        let actual = db
            .collection("notes")
            .unwrap()
            .search()
            .text("needle")
            .query_embedding(query.embedding.clone().unwrap())
            .top_k(1)
            .resource_profile(profile)
            .execute()
            .unwrap();
        let expected_score = if branch == "text" {
            // IDF and (k1 + 1) cancel in target / leader; lengths are
            // fifty ones and one two. Target is absent from B but present in V.
            let average = 52.0 / 51.0;
            let denominator = |length: f64| 1.0 + 1.2 * (1.0 - 0.75 + 0.75 * length / average);
            0.5 * denominator(1.0) / denominator(2.0) + 0.5
        } else {
            // All lengths are four. Target BM25 is M (tf4), leader/M=13/22.
            // Target is absent from V, but its true cosine0 has norm .5.
            // Its .75 beats the leaders' .5*(13/22)+.5*.9.
            0.75
        };
        assert_eq!(actual.results[0].chunk_external_id, "doc#50");
        assert!((actual.results[0].final_score - expected_score).abs() < 1e-12);
        let expected =
            acceptance::Acceptance::prepare(&db, "notes", &dataset, profile, &modes, &context)
                .unwrap();
        let hybrid = modes
            .iter()
            .position(|mode| *mode == BenchmarkQueryMode::Hybrid)
            .unwrap();
        let hits = expected.expected_hits(hybrid, 0).collect::<Vec<_>>();
        assert_eq!(hits[0].0, "doc#50");
        assert!((hits[0].1 - expected_score).abs() < 1e-12);
        for (index, mode) in modes.into_iter().enumerate() {
            run_benchmark_queries(
                &db,
                "notes",
                &dataset,
                profile,
                mode,
                "cold",
                &context,
                Some(QueryVerification {
                    expected: &expected,
                    mode: index,
                    journal: None,
                }),
            )
            .unwrap();
        }
        assert_eq!(db.temporary_disk_budget().used(), 0);
        drop((expected, actual, db, dataset, input_work));
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn source_acceptance_covers_profiles_filters_ties_and_101_query_contexts() {
    let dir = text_only_benchmark_dataset("source-acceptance-matrix");
    let mut chunks = String::new();
    let mut embeddings = String::new();
    for i in 0..71 {
        let text = [
            "recovery log",
            "recovery recovery checkpoint",
            "the and",
            "HTTPServer wal_index",
            "filter query",
        ][i % 5];
        chunks.push_str(&format!(
            "{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"doc#{i}\",\"ordinal\":{i},\"text\":\"{text}\",\"metadata\":{{\"group\":{}}}}}\n",
            i % 4,
        ));
        if i % 7 != 0 {
            embeddings.push_str(&format!(
                "{{\"chunk_external_id\":\"doc#{i}\",\"model_name\":\"m\",\"dim\":2,\"vector\":[{},{}]}}\n",
                (i % 3) as i32 - 1,
                (i % 5) as i32 - 2,
            ));
        }
    }
    fs::write(dir.join("chunks.jsonl"), &chunks).unwrap();
    fs::write(dir.join("embeddings.jsonl"), &embeddings).unwrap();
    let context = JsonOperationContext::standalone();
    let mut manifest =
        parse(&fs::read_to_string(dir.join("dataset_manifest.json")).unwrap()).unwrap();
    let hashes = manifest
        .as_object_mut()
        .unwrap()
        .get_mut("hashes")
        .unwrap()
        .as_object_mut()
        .unwrap();
    hashes.insert(
        "chunks_sha256".into(),
        JsonValue::String(crate::hash::sha256_hex(chunks.as_bytes())),
    );
    fs::write(
        dir.join("dataset_manifest.json"),
        manifest.to_compact_string(),
    )
    .unwrap();
    let (mut dataset, input_work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
    dataset.embeddings_path = Some(dir.join("embeddings.jsonl"));
    dataset.manifest.embeddings_sha256 = Some(crate::hash::sha256_hex(embeddings.as_bytes()));
    dataset.manifest.embedding_dim = 2;
    dataset.manifest.embedding_model = Some("m".into());
    dataset.queries = (0..101)
        .map(|seed| BenchmarkQuery {
            query_id: format!("query-{seed}"),
            text: Some(
                [
                    "recovery",
                    "recovery recovery log",
                    "the and",
                    "HTTPServer wal_index",
                    "absent",
                ][seed % 5]
                    .into(),
            ),
            filter: match seed % 6 {
                0 => JsonValue::object(),
                1 => parse(r#"{"chunk.group":1}"#).unwrap(),
                2 => parse(r#"{"chunk.group":{"$gte":2}}"#).unwrap(),
                3 => parse(r#"{"chunk.group":99}"#).unwrap(),
                4 => parse(r#"{"doc.absent":{"$ne":"value"}}"#).unwrap(),
                _ => parse(r#"{"doc.absent":"value"}"#).unwrap(),
            },
            embedding: Some(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![(seed % 3) as f32 - 1.0, (seed % 5) as f32 - 2.0],
            }),
            top_k: 1 + seed % 83,
            relevant_chunk_ids: BTreeSet::new(),
        })
        .collect();
    let modes = benchmark_query_modes(&dataset);
    let db = Database::create(dir.join("data.coffer"), DatabaseConfig::default()).unwrap();
    let import_manifest = write_benchmark_import_manifest(&dataset, "notes", &context).unwrap();
    db.import_manifest(import_manifest.path(), OnConflict::Replace)
        .unwrap();
    drop(import_manifest);
    let capacity = db.json_context().available_workspace_bytes();
    for profile in [
        ResourceProfile::Constrained,
        ResourceProfile::Balanced,
        ResourceProfile::Expanded,
    ] {
        acceptance::Acceptance::assert_serial_source_candidates(
            &db, "notes", &dataset, profile, &context,
        );
        let acceptance =
            acceptance::Acceptance::prepare(&db, "notes", &dataset, profile, &modes, &context)
                .unwrap();
        let serial = acceptance::Acceptance::prepare_windowed(
            &db, "notes", &dataset, profile, &modes, &context, 1,
        )
        .unwrap();
        for mode in 0..modes.len() {
            for query in 0..dataset.queries.len() {
                assert_eq!(
                    acceptance
                        .expected_hits(mode, query)
                        .map(|(id, score)| (id, score.to_bits()))
                        .collect::<Vec<_>>(),
                    serial
                        .expected_hits(mode, query)
                        .map(|(id, score)| (id, score.to_bits()))
                        .collect::<Vec<_>>(),
                );
            }
        }
        drop(serial);
        assert_eq!(db.temporary_disk_budget().used(), 0);
        for (mode_index, &mode) in modes.iter().enumerate() {
            run_benchmark_queries(
                &db,
                "notes",
                &dataset,
                profile,
                mode,
                "cold",
                &context,
                Some(QueryVerification {
                    expected: &acceptance,
                    mode: mode_index,
                    journal: None,
                }),
            )
            .unwrap();
        }
    }
    let tail = dataset.queries.split_off(5);
    let available = context.available_workspace_bytes();
    let (result, scans) = source_probe::measure(None, || {
        acceptance::Acceptance::prepare(
            &db,
            "notes",
            &dataset,
            ResourceProfile::Balanced,
            &modes,
            &context,
        )
    });
    // Two bounded query groups: each uses two recall chunk passes plus
    // one text-completion pass; only the first needs vector completion.
    assert_eq!(scans.opened, [1, 6, 3]);
    assert_eq!(scans.verified, scans.opened);
    assert_eq!(
        scans.bytes,
        [
            fs::metadata(&dataset.documents_path).unwrap().len() as usize,
            chunks.len() * 6,
            embeddings.len() * 3,
        ]
    );
    drop(result.unwrap());
    assert_eq!(context.available_workspace_bytes(), available);
    for (kind, ordinal, input, opened, verified) in [
        (1, 2, chunks.as_str(), [1, 2, 0], [1, 1, 0]),
        (2, 1, embeddings.as_str(), [1, 2, 1], [1, 2, 0]),
        (1, 3, chunks.as_str(), [1, 3, 1], [1, 2, 1]),
        (2, 2, embeddings.as_str(), [1, 3, 2], [1, 3, 1]),
    ] {
        for fault in [
            source_probe::Fault::CorruptAt(input.len() - 1),
            source_probe::Fault::ReadAfter(input.find('\n').unwrap() + 1),
        ] {
            let (result, scans) = source_probe::measure(Some((kind, ordinal, fault)), || {
                acceptance::Acceptance::prepare(
                    &db,
                    "notes",
                    &dataset,
                    ResourceProfile::Balanced,
                    &modes,
                    &context,
                )
            });
            let error = result.err().unwrap();
            assert_eq!(
                error.code(),
                match fault {
                    source_probe::Fault::CorruptAt(_) => ErrorCode::ChecksumMismatch,
                    source_probe::Fault::ReadAfter(_) => ErrorCode::IoError,
                }
            );
            assert_eq!(scans.injected, 1);
            assert_eq!(scans.opened, opened);
            assert_eq!(scans.verified, verified);
            assert_eq!(context.available_workspace_bytes(), available);
            assert_eq!(db.temporary_disk_budget().used(), 0);
        }
    }
    let filters = dataset
        .queries
        .iter_mut()
        .map(|query| {
            std::mem::replace(
                &mut query.filter,
                parse(r#"{"chunk.group":{"$gte":0}}"#).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let budget = db.temporary_disk_budget();
    let lookup_bytes = (0..71)
        .map(|index| 20 + format!("doc#{index}").len() as u64 + 16)
        .sum::<u64>();
    let (held_file, mut held_disk) = crate::temporary::create_budgeted_temporary_file_in(
        &dir,
        "acceptance-busy-disk",
        "tmp",
        budget.clone(),
    )
    .unwrap();
    held_disk
        .reserve_disk(budget.limit_bytes() - lookup_bytes)
        .unwrap();
    let used = budget.used();
    let (result, scans) = source_probe::measure(None, || {
        acceptance::Acceptance::prepare(
            &db,
            "notes",
            &dataset,
            ResourceProfile::Balanced,
            &modes,
            &context,
        )
    });
    let acceptance = result.unwrap();
    // Two refused multi-query groups read their initial statistics; the
    // five successful serial groups all complete text, and three complete vectors.
    assert_eq!(scans.opened, [1, 17, 8]);
    assert_eq!(scans.verified, scans.opened);
    assert_eq!(budget.used(), used);
    drop(acceptance);
    assert_eq!(context.available_workspace_bytes(), available);
    drop(held_file);
    drop(held_disk);
    assert_eq!(budget.used(), 0);
    for (query, filter) in dataset.queries.iter_mut().zip(filters) {
        query.filter = filter;
    }
    dataset.queries.extend(tail);
    let mut busy = context.reservation();
    busy.grow(
        capacity - input_work.reserved_bytes(),
        JsonInputKind::BenchmarkArtifact,
    )
    .unwrap();
    assert!(acceptance::Acceptance::prepare(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        &modes,
        &context
    )
    .is_err());
    assert_eq!(db.temporary_disk_budget().used(), 0);
    drop(busy);
    fs::write(dir.join("chunks.jsonl"), format!("{chunks}\n")).unwrap();
    let error = acceptance::Acceptance::prepare(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        &modes,
        &context,
    )
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
    assert_eq!(db.temporary_disk_budget().used(), 0);
    fs::write(dir.join("chunks.jsonl"), chunks).unwrap();
    let acceptance = acceptance::Acceptance::prepare(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        &modes,
        &context,
    )
    .unwrap();
    let collection = db.collection("notes").unwrap();
    let document = collection.get_document_by_external_id("doc").unwrap();
    collection.delete_document(&document.document_id).unwrap();
    let error = run_benchmark_queries(
        &db,
        "notes",
        &dataset,
        ResourceProfile::Balanced,
        modes[0],
        "cold",
        &context,
        Some(QueryVerification {
            expected: &acceptance,
            mode: 0,
            journal: None,
        }),
    )
    .err()
    .unwrap();
    assert_eq!(
        detail_value(&error, "reason"),
        Some("benchmark_acceptance_mismatch")
    );
    assert_eq!(detail_value(&error, "query_id"), Some("query-0"));
    drop((acceptance, collection, db, dataset, input_work));
    context
        .reservation()
        .grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn benchmark_profile_controls_import_batches_for_new_and_existing_databases() {
    let dir = text_only_benchmark_dataset("benchmark-runtime-profile");
    let chunk_count = 501usize;
    let mut rows = String::new();
    for index in 0..chunk_count {
        let chunk = object_from_pairs([
            ("document_external_id", JsonValue::String("doc".into())),
            (
                "chunk_external_id",
                JsonValue::String(format!("doc#{}", index + 1)),
            ),
            ("ordinal", JsonValue::Number(index as f64)),
            ("text", JsonValue::String("benchmark query".into())),
            (
                "metadata",
                object_from_pairs([("lang", JsonValue::String("rust".into()))]),
            ),
        ]);
        rows.push_str(&chunk.to_compact_string());
        rows.push('\n');
    }
    fs::write(dir.join("chunks.jsonl"), &rows).unwrap();
    let manifest_path = dir.join("dataset_manifest.json");
    let mut manifest = parse(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest
        .as_object_mut()
        .unwrap()
        .get_mut("hashes")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert(
            "chunks_sha256".into(),
            JsonValue::String(crate::hash::sha256_hex(rows.as_bytes())),
        );
    fs::write(&manifest_path, manifest.to_compact_string()).unwrap();
    for profile in [
        ResourceProfile::Constrained,
        ResourceProfile::Balanced,
        ResourceProfile::Expanded,
    ] {
        for durability in [Durability::Commit, Durability::Fast] {
            for existing in [false, true] {
                let label = format!(
                    "{}-{}-{existing}",
                    profile.as_json_str(),
                    durability.as_str()
                );
                let path = dir.join(format!("{label}.coffer"));
                if existing {
                    Database::create(&path, DatabaseConfig::default())
                        .unwrap()
                        .close()
                        .unwrap();
                }
                let output = run_benchmark(BenchmarkOptions {
                    path: path.to_string_lossy().into_owned(),
                    dataset_path: dir.to_string_lossy().into_owned(),
                    collection: "notes".into(),
                    profile,
                    durability,
                    result_file: None,
                    require_baseline: false,
                })
                .unwrap();
                let scan = crate::storage::scan_file(&path).unwrap();
                // One collection record and one document precede the chunk records.
                let records = chunk_count + 2;
                let batch_limit = profile.params().import_batch_chunks;
                assert_eq!(scan.commits, records.div_ceil(batch_limit), "{label}");
                assert_eq!(
                    output
                        .value
                        .get("import_context")
                        .unwrap()
                        .get("batch_size")
                        .unwrap()
                        .as_f64(),
                    Some(chunk_count.min(batch_limit) as f64),
                    "actual largest committed chunk batch: {label}",
                );
                let db = Database::open_read_only(&path).unwrap();
                assert_eq!(
                    db.collection("notes").unwrap().stats().unwrap().chunk_count,
                    chunk_count
                );
                for run in output.value.get("runs").unwrap().as_array().unwrap() {
                    let row = &run.get("per_query").unwrap().as_array().unwrap()[0];
                    let context = row.get("query_context").unwrap();
                    assert_eq!(
                        context.get("profile").unwrap().as_str(),
                        Some(profile.as_json_str())
                    );
                    assert_eq!(
                        context
                            .get("transient_rebuild")
                            .unwrap()
                            .get("budget_mb")
                            .unwrap()
                            .as_f64(),
                        Some(profile.params().cache_budget_mb as f64),
                    );
                }
                db.close().unwrap();
                let replaced = run_benchmark(BenchmarkOptions {
                    path: path.to_string_lossy().into_owned(),
                    dataset_path: dir.to_string_lossy().into_owned(),
                    collection: "notes".into(),
                    profile,
                    durability,
                    result_file: None,
                    require_baseline: false,
                })
                .unwrap();
                let mut chunks_by_batch = BTreeMap::<u64, usize>::new();
                for record in crate::storage::scan_file(&path).unwrap().records {
                    if record.record_ref.batch_id > scan.last_valid_commit
                        && matches!(record.payload, crate::storage::RecordPayload::ChunkPut(_))
                    {
                        *chunks_by_batch
                            .entry(record.record_ref.batch_id)
                            .or_default() += 1;
                    }
                }
                assert_eq!(chunks_by_batch.values().sum::<usize>(), chunk_count);
                assert_eq!(
                    replaced
                        .value
                        .get("import_context")
                        .unwrap()
                        .get("batch_size")
                        .unwrap()
                        .as_f64(),
                    chunks_by_batch.values().max().map(|count| *count as f64),
                    "replacement tombstones consume the record cap: {label}",
                );
            }
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sqlite_baseline_reports_share_input_budget_and_release_on_failure() {
    let dir = text_only_benchmark_dataset("cofferdb-sqlite-shared-work");
    let context = JsonOperationContext::standalone();
    let (dataset, input_work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
    let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
    let mut busy = context.reservation();
    busy.grow(
        limit - input_work.reserved_bytes(),
        JsonInputKind::BenchmarkArtifact,
    )
    .unwrap();
    let error = read_sqlite_baseline_data(&dataset, &context).err().unwrap();
    assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        detail_value(&error, "json_limit_reason"),
        Some("work_budget_exceeded")
    );
    drop(busy);
    drop(read_sqlite_baseline_data(&dataset, &context).unwrap());
    if sqlite3_fts5_available() {
        for profile in [
            ResourceProfile::Constrained,
            ResourceProfile::Balanced,
            ResourceProfile::Expanded,
        ] {
            let (runs, work) = run_sqlite_baseline(&dataset, profile, &context).unwrap();
            assert_eq!(runs.len(), 2);
            for (run, state) in runs.iter().zip(["cold", "warm"]) {
                assert_eq!(run.get("cache_state").unwrap().as_str(), Some(state));
                let row = &run.get("per_query").unwrap().as_array().unwrap()[0];
                assert_eq!(
                    row.get("ranked_chunk_ids").unwrap(),
                    &JsonValue::Array(vec![JsonValue::String("doc#1".into())])
                );
            }
            let runs = JsonValue::Array(runs);
            let expected = runs
                .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), expected.reserved_bytes());
            drop(expected);
            let mut busy = context.reservation();
            busy.grow(
                limit - input_work.reserved_bytes() - work.reserved_bytes(),
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
            assert!(run_sqlite_baseline(&dataset, profile, &context).is_err());
            let (outcome, failed_work) = benchmark_baseline_outcome(&dataset, profile, &context);
            assert_eq!(outcome.status, "not_run");
            assert!(outcome.results.is_empty());
            assert_eq!(failed_work.reserved_bytes(), 0);
            drop(busy);
            drop(runs);
            drop(work);
            drop(run_sqlite_baseline(&dataset, profile, &context).unwrap());
        }
    }
    drop(dataset);
    drop(input_work);
    context
        .reservation()
        .grow(limit, JsonInputKind::BenchmarkArtifact)
        .unwrap();
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sqlite_query_context_matches_contract_under_shared_output_budget() {
    for seed in 0..100 {
        let profile = [
            ResourceProfile::Balanced,
            ResourceProfile::Constrained,
            ResourceProfile::Expanded,
        ][seed % 3];
        let query = BenchmarkQuery {
            query_id: "q".into(),
            text: Some("rust".into()),
            filter: if seed % 2 == 0 {
                JsonValue::object()
            } else {
                parse(r#"{"chunk.n":1}"#).unwrap()
            },
            top_k: seed + 1,
            embedding: None,
            relevant_chunk_ids: BTreeSet::new(),
        };
        for mode in [
            BenchmarkQueryMode::TextBm25,
            BenchmarkQueryMode::VectorExact,
            BenchmarkQueryMode::Hybrid,
        ] {
            let context = JsonOperationContext::standalone();
            let mut work = context.reservation();
            let params = profile.params();
            let limits = CandidateLimits {
                bm25: if mode.has_text() {
                    params.bm25_candidates.max(query.top_k)
                } else {
                    0
                },
                vector: if mode.has_vector() {
                    params.vector_candidates.max(query.top_k)
                } else {
                    0
                },
                fusion: if mode.has_vector() {
                    params.fusion_candidates.max(query.top_k)
                } else {
                    0
                },
            };
            let stats = BenchmarkExecutionStats {
                vector_scan_count: seed,
                distance_ops: seed * 768,
                fusion_candidate_count: seed * 2,
            };
            let value = sqlite_baseline_query_context(
                &query,
                profile,
                mode,
                limits,
                stats,
                &JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact),
            )
            .unwrap();
            let mut expected = benchmark_contract_query_context_for_mode(mode.as_str());
            for (key, value) in [
                ("tokenizer", JsonValue::Null),
                (
                    "index_source",
                    JsonValue::String(sqlite_index_source(mode).into()),
                ),
                ("top_k", JsonValue::Number(query.top_k as f64)),
                ("profile", JsonValue::String(profile.as_json_str().into())),
                ("filter_present", JsonValue::Bool(seed % 2 != 0)),
                ("execution_stats", stats.to_json()),
                (
                    "candidate_limits",
                    object_from_pairs([
                        ("bm25", JsonValue::Number(limits.bm25 as f64)),
                        ("vector", JsonValue::Number(limits.vector as f64)),
                        ("fusion", JsonValue::Number(limits.fusion as f64)),
                    ]),
                ),
            ] {
                insert_object_field(&mut expected, key, value);
            }
            insert_object_field(
                expected
                    .as_object_mut()
                    .unwrap()
                    .get_mut("transient_rebuild")
                    .unwrap(),
                "budget_mb",
                JsonValue::Number(0.0),
            );
            assert_eq!(value, expected);
            let expected_work = value
                .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), expected_work.reserved_bytes());
            drop(expected_work);
            drop(value);
            drop(work);
            let mut busy = context.reservation();
            busy.grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
            let mut work = context.reservation();
            let error = sqlite_baseline_query_context(
                &query,
                profile,
                mode,
                limits,
                stats,
                &JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact),
            )
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            assert_eq!(work.reserved_bytes(), 0);
        }
    }
}

#[test]
fn benchmark_rejects_outputs_over_database_and_all_dataset_inputs_before_import() {
    let dir = text_only_benchmark_dataset("cofferdb-benchmark-protected-output");
    let database = dir.join("benchmark.coffer");
    let inputs = [
        "dataset_manifest.json",
        "documents.jsonl",
        "chunks.jsonl",
        "queries.jsonl",
        "qrels.jsonl",
    ];
    let original = inputs.map(|name| (name, fs::read(dir.join(name)).unwrap()));
    for existing in [false, true] {
        if existing {
            drop(Database::create(&database, DatabaseConfig::default()).unwrap());
        }
        let before = fs::read(&database).ok();
        let files = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<BTreeSet<_>>();
        let lock = fs::read(crate::lock::lock_path(&database)).ok();
        for flag in ["--output", "--result-file"] {
            for name in inputs
                .into_iter()
                .chain(["benchmark.coffer", "benchmark.coffer.lock"])
            {
                let error = run(vec![
                    "coffer".into(),
                    "benchmark".into(),
                    database.display().to_string(),
                    "--dataset".into(),
                    dir.display().to_string(),
                    flag.into(),
                    dir.join(name).display().to_string(),
                ])
                .unwrap_err();
                assert_eq!(error.code(), ErrorCode::PermissionDenied, "{name}");
                assert_eq!(fs::read(&database).ok(), before);
                for (name, bytes) in &original {
                    assert_eq!(&fs::read(dir.join(name)).unwrap(), bytes);
                }
                assert_eq!(fs::read(crate::lock::lock_path(&database)).ok(), lock);
                assert_eq!(
                    fs::read_dir(&dir)
                        .unwrap()
                        .map(|entry| entry.unwrap().file_name())
                        .collect::<BTreeSet<_>>(),
                    files
                );
            }
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn benchmark_runs_cold_and_warm_queries() {
    let dir = text_only_benchmark_dataset("cofferdb-cli-benchmark");
    let db_path = dir.join("benchmark.coffer");
    let result_path = dir.join("benchmark_result.json");
    let result_path_str = result_path.display().to_string();
    let output = run(vec![
        "coffer".into(),
        "benchmark".into(),
        db_path.display().to_string(),
        "--dataset".into(),
        dir.display().to_string(),
        "--profile".into(),
        "constrained".into(),
        "--durability".into(),
        "fast".into(),
        "--output".into(),
        result_path_str.clone(),
    ])
    .unwrap();
    let json = parse(&output).unwrap();
    let result = json.get("result").and_then(JsonValue::as_object).unwrap();
    assert_eq!(
        result.get("profile").and_then(JsonValue::as_str),
        Some("constrained")
    );
    assert_eq!(
        result.get("durability").and_then(JsonValue::as_str),
        Some("fast")
    );
    assert_eq!(
        result.get("result_file").and_then(JsonValue::as_str),
        Some(result_path_str.as_str())
    );
    let saved = parse(&std::fs::read_to_string(&result_path).unwrap()).unwrap();
    assert_eq!(
        saved.get("profile").and_then(JsonValue::as_str),
        Some("constrained")
    );
    assert_eq!(
        result.get("dataset_id").and_then(JsonValue::as_str),
        Some("bench")
    );
    let dataset_identity = result
        .get("dataset")
        .and_then(JsonValue::as_object)
        .unwrap();
    assert_eq!(
        dataset_identity
            .get("query_embedding_source")
            .and_then(JsonValue::as_str),
        Some("none")
    );
    assert_eq!(
        result
            .get("profile_params")
            .and_then(|params| params.get("fusion_candidates"))
            .and_then(JsonValue::as_f64),
        Some(50.0)
    );
    assert_eq!(
        result
            .get("engine_params")
            .and_then(|params| params.get("text_index"))
            .and_then(|text| text.get("bm25"))
            .and_then(|bm25| bm25.get("k1"))
            .and_then(JsonValue::as_f64),
        Some(crate::index::text::BM25_K1)
    );
    assert_eq!(
        result
            .get("engine_params")
            .and_then(|params| params.get("text_index"))
            .and_then(|text| text.get("tokenizer"))
            .and_then(|tokenizer| tokenizer.get("name"))
            .and_then(JsonValue::as_str),
        Some(crate::index::text::DEFAULT_TOKENIZER_NAME)
    );
    assert_eq!(
        result.get("status").and_then(JsonValue::as_str),
        Some("completed_cofferdb_runs")
    );
    assert!(result
        .get("git_commit")
        .and_then(JsonValue::as_str)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false));
    let import_context = result
        .get("import_context")
        .and_then(JsonValue::as_object)
        .unwrap();
    assert!(import_context
        .get("summary")
        .and_then(JsonValue::as_object)
        .is_some());
    assert_eq!(
        import_context
            .get("manifest_path")
            .and_then(JsonValue::as_str),
        Some("<generated-import-manifest>")
    );
    let db_path_label = db_path.display().to_string();
    assert_eq!(
        import_context
            .get("database_path")
            .and_then(JsonValue::as_str),
        Some(db_path_label.as_str())
    );
    assert_eq!(
        import_context
            .get("on_conflict")
            .and_then(JsonValue::as_str),
        Some("replace")
    );
    let runs = result.get("runs").and_then(JsonValue::as_array).unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(
        runs[0].get("cache_state").and_then(JsonValue::as_str),
        Some("cold")
    );
    assert_eq!(
        runs[0].get("query_mode").and_then(JsonValue::as_str),
        Some("text_bm25")
    );
    assert_eq!(
        runs[1].get("cache_state").and_then(JsonValue::as_str),
        Some("warm")
    );
    assert_eq!(
        runs[1]
            .get("metrics")
            .and_then(|metrics| metrics.get("recall_at_10"))
            .and_then(JsonValue::as_f64),
        Some(1.0)
    );
    let baseline_configs = result
        .get("baseline_configs")
        .and_then(JsonValue::as_array)
        .unwrap();
    assert_eq!(
        baseline_configs[0]
            .get("requires_query_embedding")
            .and_then(JsonValue::as_bool),
        Some(false)
    );
    assert_eq!(
        baseline_configs[0].get("mode").and_then(JsonValue::as_str),
        Some("local_cli")
    );
    assert_eq!(
        baseline_configs[0].get("top_k").and_then(JsonValue::as_f64),
        Some(10.0)
    );
    assert_eq!(
        baseline_configs[0]
            .get("cache_states")
            .and_then(JsonValue::as_array)
            .map(|states| states.len()),
        Some(2)
    );
    assert_eq!(
        baseline_configs[0]
            .get("config")
            .and_then(|config| config.get("fts_tokenizer"))
            .and_then(JsonValue::as_str),
        Some("default")
    );
    let baseline_status = result
        .get("baseline_status")
        .and_then(JsonValue::as_str)
        .unwrap();
    let baseline_results = result
        .get("baseline_results")
        .and_then(JsonValue::as_array)
        .unwrap();
    if sqlite3_fts5_available() {
        assert_eq!(
            baseline_status,
            "completed",
            "{:?}",
            result.get("baseline_status_reason")
        );
        assert_eq!(baseline_results.len(), 2);
        for run in baseline_results {
            assert_eq!(
                run.get("metrics")
                    .and_then(|metrics| metrics.get("memory_peak_mb")),
                Some(&JsonValue::Null)
            );
        }
        assert!(baseline_configs[0]
            .get("checked_at")
            .and_then(JsonValue::as_str)
            .map(|value| value.len() == 10)
            .unwrap_or(false));
        assert_eq!(
            baseline_configs[0]
                .get("config")
                .and_then(|config| config.get("status"))
                .and_then(JsonValue::as_str),
            Some("completed")
        );
        assert_eq!(
            baseline_results[0]
                .get("cache_state")
                .and_then(JsonValue::as_str),
            Some("cold")
        );
        assert_eq!(
            baseline_results[1]
                .get("cache_state")
                .and_then(JsonValue::as_str),
            Some("warm")
        );
        assert_eq!(
            baseline_results[1]
                .get("per_query")
                .and_then(JsonValue::as_array)
                .and_then(|queries| queries.first())
                .and_then(|query| query.get("hit"))
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            baseline_results[1]
                .get("per_query")
                .and_then(JsonValue::as_array)
                .and_then(|queries| queries.first())
                .and_then(|query| query.get("ranked_chunk_ids"))
                .and_then(JsonValue::as_array)
                .and_then(|ids| ids.first())
                .and_then(JsonValue::as_str),
            Some("doc#1")
        );
    } else {
        assert_eq!(baseline_status, "not_run");
        assert!(baseline_results.is_empty());
        assert_eq!(
            baseline_configs[0].get("checked_at"),
            Some(&JsonValue::Null)
        );
        assert_eq!(
            baseline_configs[0]
                .get("config")
                .and_then(|config| config.get("status"))
                .and_then(JsonValue::as_str),
            Some("not_run")
        );
    }
    let per_query = runs[1]
        .get("per_query")
        .and_then(JsonValue::as_array)
        .unwrap();
    assert_eq!(
        per_query[0].get("hit").and_then(JsonValue::as_bool),
        Some(true)
    );

    let output = run(vec![
        "coffer".into(),
        "benchmark".into(),
        "notes.coffer".into(),
        "--profile".into(),
        "help".into(),
    ])
    .unwrap();
    let json = parse(&output).unwrap();
    let result = json.get("result").and_then(JsonValue::as_object).unwrap();
    assert!(matches!(result.get("profiles"), Some(JsonValue::Array(_))));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn benchmark_sidecar_queries_report_vector_stats() {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "cofferdb-cli-benchmark-sidecar-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("benchmark-sidecar.coffer");
    std::fs::write(
        dir.join("documents.jsonl"),
        r#"{"external_id":"doc","metadata":{}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("chunks.jsonl"),
        r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"semantic wal recovery","metadata":{}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("embeddings.jsonl"),
        r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("queries.jsonl"),
        r#"{"query_id":"q1","text":"semantic wal","top_k":10}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("qrels.jsonl"),
        r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("query_embeddings.jsonl"),
        r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
    )
    .unwrap();
    let documents_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("documents.jsonl")).unwrap());
    let chunks_sha256 = crate::hash::sha256_hex(&std::fs::read(dir.join("chunks.jsonl")).unwrap());
    let embeddings_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("embeddings.jsonl")).unwrap());
    let queries_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("queries.jsonl")).unwrap());
    let qrels_sha256 = crate::hash::sha256_hex(&std::fs::read(dir.join("qrels.jsonl")).unwrap());
    let query_embeddings_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("query_embeddings.jsonl")).unwrap());
    std::fs::write(
        dir.join("dataset_manifest.json"),
        format!(
            concat!(
                r#"{{"schema_version":"0.1","dataset_id":"bench-sidecar","version":"0.1","#,
                r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                r#"}}"#
            ),
            documents_sha256,
            chunks_sha256,
            embeddings_sha256,
            queries_sha256,
            qrels_sha256,
            query_embeddings_sha256
        ),
    )
    .unwrap();

    let output = run(vec![
        "coffer".into(),
        "benchmark".into(),
        db_path.display().to_string(),
        "--dataset".into(),
        dir.display().to_string(),
    ])
    .unwrap();
    let json = parse(&output).unwrap();
    let envelope_warnings = json.get("warnings").and_then(JsonValue::as_array).unwrap();
    let result = json.get("result").and_then(JsonValue::as_object).unwrap();
    let result_warnings = result
        .get("warnings")
        .and_then(JsonValue::as_array)
        .unwrap();
    let metrics = result
        .get("metrics")
        .and_then(JsonValue::as_object)
        .unwrap();
    assert_eq!(
        metrics.get("vector_scan_count").and_then(JsonValue::as_f64),
        Some(1.0)
    );
    assert_eq!(
        metrics.get("distance_ops").and_then(JsonValue::as_f64),
        Some(2.0)
    );
    let baseline_configs = result
        .get("baseline_configs")
        .and_then(JsonValue::as_array)
        .unwrap();
    assert_eq!(
        baseline_configs[0].get("name").and_then(JsonValue::as_str),
        Some("sqlite-fts5-sqlite-vec")
    );
    assert_eq!(
        baseline_configs[0]
            .get("requires_query_embedding")
            .and_then(JsonValue::as_bool),
        Some(true)
    );
    assert_eq!(
        baseline_configs[0]
            .get("config")
            .and_then(|config| config.get("vector_extension"))
            .and_then(JsonValue::as_str),
        Some("sqlite-vec")
    );
    assert_eq!(
        baseline_configs[0]
            .get("config")
            .and_then(|config| config.get("fusion"))
            .and_then(JsonValue::as_str),
        Some("weighted_sum")
    );
    assert_eq!(
        baseline_configs[0]
            .get("config")
            .and_then(|config| config.get("filter"))
            .and_then(JsonValue::as_str),
        Some("metadata_filter_before_vector_scoring")
    );
    let baseline_results = result
        .get("baseline_results")
        .and_then(JsonValue::as_array)
        .unwrap();
    if sqlite3_fts5_available() && sqlite3_sqlite_vec_available() {
        assert!(envelope_warnings.is_empty());
        assert!(result_warnings.is_empty());
        assert_eq!(
            result.get("baseline_status").and_then(JsonValue::as_str),
            Some("completed")
        );
        assert_eq!(baseline_results.len(), 6);
        for run in baseline_results {
            assert_eq!(
                run.get("metrics")
                    .and_then(|metrics| metrics.get("memory_peak_mb")),
                Some(&JsonValue::Null)
            );
        }
        assert_eq!(result.get("baseline_status_reason"), Some(&JsonValue::Null));
    } else {
        assert_eq!(
            envelope_warnings[0].get("code").and_then(JsonValue::as_str),
            Some("benchmark_baseline_not_run")
        );
        assert_eq!(
            result_warnings[0].get("code").and_then(JsonValue::as_str),
            Some("benchmark_baseline_not_run")
        );
        assert_eq!(
            result.get("baseline_status").and_then(JsonValue::as_str),
            Some("not_run")
        );
        assert!(baseline_results.is_empty());
        assert_eq!(
            baseline_configs[0].get("checked_at"),
            Some(&JsonValue::Null)
        );
        assert_eq!(
            baseline_configs[0]
                .get("config")
                .and_then(|config| config.get("status"))
                .and_then(JsonValue::as_str),
            Some("not_run")
        );
        let baseline_status_reason = result
            .get("baseline_status_reason")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        if sqlite3_fts5_available() {
            assert!(
                baseline_status_reason.contains("sqlite-vec"),
                "{baseline_status_reason}"
            );
        } else {
            assert!(
                baseline_status_reason.contains("FTS5"),
                "{baseline_status_reason}"
            );
        }
    }
    let runs = result.get("runs").and_then(JsonValue::as_array).unwrap();
    assert_eq!(runs.len(), 6);
    let hybrid_warm = runs
        .iter()
        .find(|run| {
            run.get("query_mode").and_then(JsonValue::as_str) == Some("hybrid")
                && run.get("cache_state").and_then(JsonValue::as_str) == Some("warm")
        })
        .unwrap();
    let per_query = hybrid_warm
        .get("per_query")
        .and_then(JsonValue::as_array)
        .unwrap();
    assert_eq!(
        per_query[0]
            .get("query_context")
            .and_then(|context| context.get("embedding_present"))
            .and_then(JsonValue::as_bool),
        Some(true)
    );
    assert_eq!(
        per_query[0]
            .get("execution_stats")
            .and_then(|stats| stats.get("vector_scan_count"))
            .and_then(JsonValue::as_f64),
        Some(1.0)
    );
    assert_eq!(
        per_query[0]
            .get("ranked_chunk_ids")
            .and_then(JsonValue::as_array)
            .and_then(|ids| ids.first())
            .and_then(JsonValue::as_str),
        Some("doc#1")
    );
    let required_baseline = run(vec![
        "coffer".into(),
        "benchmark".into(),
        db_path.display().to_string(),
        "--dataset".into(),
        dir.display().to_string(),
        "--require-baseline".into(),
    ]);
    if sqlite3_fts5_available() && sqlite3_sqlite_vec_available() {
        assert!(required_baseline.is_ok());
        let _ = std::fs::remove_dir_all(dir);
        return;
    }
    let err = required_baseline.unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_status")
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn sqlite_baseline_rejects_changed_artifacts_before_producing_comparison_data() {
    let dir = text_only_benchmark_dataset("sqlite-verified-inputs");
    let mut dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
    let embeddings = concat!(
        r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1,0]}"#,
        "\n",
        r#"{"chunk_external_id":"doc#2","model_name":"m","dim":2,"vector":[0,1]}"#,
        "\n"
    );
    let embedding_path = dir.join("embeddings.jsonl");
    fs::write(&embedding_path, embeddings).unwrap();
    dataset.embeddings_path = Some(embedding_path.clone());
    dataset.manifest.embeddings_sha256 = Some(crate::hash::sha256_hex(embeddings.as_bytes()));
    let paths = [
        dataset.documents_path.clone(),
        dataset.chunks_path.clone(),
        embedding_path,
    ];
    let kinds = ["documents", "chunks", "embeddings"];
    let originals = paths
        .iter()
        .map(|path| fs::read(path).unwrap())
        .collect::<Vec<_>>();
    let context = JsonOperationContext::standalone();
    drop(read_sqlite_baseline_data(&dataset, &context).unwrap());
    for seed in 0..100 {
        let index = seed % paths.len();
        let mut changed = originals[index].clone();
        // Byte identity includes whitespace, including bytes beyond the reader's buffer.
        changed.extend(vec![b' '; seed * 211 + 1]);
        fs::write(&paths[index], &changed).unwrap();
        let error = read_sqlite_baseline_data(&dataset, &context)
            .err()
            .expect("baseline must reject changed input even when its JSON remains valid");
        assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        assert_eq!(detail_value(&error, "artifact_kind"), Some(kinds[index]));
        assert_eq!(
            detail_value(&error, "expected_sha256"),
            Some(crate::hash::sha256_hex(&originals[index]).as_str())
        );
        assert_eq!(
            detail_value(&error, "actual_sha256"),
            Some(crate::hash::sha256_hex(&changed).as_str())
        );
        assert_eq!(detail_value(&error, "path"), paths[index].to_str());
        fs::write(&paths[index], &originals[index]).unwrap();
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
    }
    let data = read_sqlite_baseline_data(&dataset, &context).unwrap();
    assert_eq!(data.catalog.chunks.len(), 2);
    let mut count = 0;
    visit_sqlite_baseline_embeddings(&dataset, &context, |_| {
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 2);
    drop(data);
    dataset.manifest.embeddings_sha256 = None;
    let error = visit_sqlite_baseline_embeddings(&dataset, &context, |_| Ok(())).unwrap_err();
    assert_eq!(error.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        detail_value(&error, "field"),
        Some("hashes.embeddings_sha256")
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn verified_baseline_input_hash_is_bound_to_the_stream_being_consumed() {
    let (file, original) = create_temporary_file("baseline-stream-hash", "jsonl").unwrap();
    drop(file);
    let before = b"original bytes";
    let after = b"other snapshot";
    fs::write(original.path(), before).unwrap();
    let expected = crate::hash::sha256_hex(before);
    let (file, replacement) = create_temporary_file("baseline-replacement", "jsonl").unwrap();
    drop(file);
    fs::write(replacement.path(), after).unwrap();
    let read = read_verified_baseline_input(original.path(), "chunks", &expected, |source| {
        fs::rename(replacement.path(), original.path())?;
        let mut bytes = Vec::new();
        source.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
    .unwrap();
    assert_eq!(read, before);
    assert_eq!(fs::read(original.path()).unwrap(), after);

    fs::write(original.path(), before).unwrap();
    let error = read_verified_baseline_input(original.path(), "chunks", &expected, |source| {
        let mut first = [0];
        source.read_exact(&mut first)?;
        // Preserve the consumed prefix while changing bytes that have not been read.
        fs::write(original.path(), after)?;
        assert_eq!(first[0], after[0]);
        std::io::copy(source, &mut std::io::sink())?;
        Ok(())
    })
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
    assert_eq!(
        detail_value(&error, "actual_sha256"),
        Some(crate::hash::sha256_hex(after).as_str())
    );
}

#[test]
fn sqlite_baseline_releases_consumed_long_document_references() {
    let (mut documents_file, documents_path) =
        create_temporary_file("sqlite-long-reference-documents", "jsonl").unwrap();
    let (mut chunks_file, chunks_path) =
        create_temporary_file("sqlite-long-reference-chunks", "jsonl").unwrap();
    let external_id = "d".repeat(600 * 1024);
    writeln!(
        documents_file,
        "{{\"external_id\":\"{external_id}\",\"metadata\":{{}}}}"
    )
    .unwrap();
    for index in 0..64 {
        writeln!(chunks_file, "{{\"document_external_id\":\"{external_id}\",\"chunk_external_id\":\"chunk-{index}\",\"ordinal\":{index},\"text\":\"rust\",\"metadata\":{{}}}}").unwrap();
    }
    drop((documents_file, chunks_file));
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    let mut work = context.catalog_reservation();
    let documents = read_sqlite_baseline_documents(
        documents_path.path(),
        File::open(documents_path.path()).unwrap(),
        &context,
        &mut work,
    )
    .unwrap();
    let (catalog, ids) = read_sqlite_baseline_chunks(
        chunks_path.path(),
        File::open(chunks_path.path()).unwrap(),
        &documents,
        &context,
        &mut work,
    )
    .unwrap();
    assert_eq!(catalog.chunks.len(), 64);
    assert_eq!(ids.len(), 64);
    assert_eq!(
        work.reserved_bytes(),
        baseline_document_bytes(&documents) + baseline_chunk_bytes(&catalog, &ids)
    );
    assert_eq!(context.available_workspace_bytes(), available);
    drop((catalog, ids, documents, work));
    assert_eq!(context.available_workspace_bytes(), available);
}

fn baseline_document_bytes(documents: &BTreeMap<String, Box<Document>>) -> usize {
    btree_workspace::<String, Box<Document>>(documents.len())
        + documents
            .iter()
            .map(|(key, document)| {
                std::mem::size_of::<Document>()
                    + key.capacity()
                    + document.document_id.capacity()
                    + document.collection_id.capacity()
                    + document.external_id.capacity()
                    + document.title.as_ref().map_or(0, String::capacity)
                    + document.source_uri.as_ref().map_or(0, String::capacity)
                    + document
                        .metadata
                        .reserve_retained_in(
                            JsonInputKind::DocumentMetadata,
                            &JsonOperationContext::standalone(),
                        )
                        .unwrap()
                        .reserved_bytes()
            })
            .sum::<usize>()
}

fn baseline_chunk_bytes(catalog: &Catalog, ids: &BTreeMap<String, String>) -> usize {
    btree_workspace::<String, Box<Chunk>>(catalog.chunks.len())
        + btree_workspace::<String, String>(ids.len())
        + ids
            .iter()
            .map(|(key, id)| key.capacity() + id.capacity())
            .sum::<usize>()
        + catalog
            .chunks
            .iter()
            .map(|(key, chunk)| {
                std::mem::size_of::<Chunk>()
                    + key.capacity()
                    + chunk.chunk_id.capacity()
                    + chunk.document_id.capacity()
                    + chunk.collection_id.capacity()
                    + chunk.chunk_external_id.capacity()
                    + chunk.text.capacity()
                    + chunk.text_hash.capacity()
                    + chunk
                        .metadata
                        .reserve_retained_in(
                            JsonInputKind::ChunkMetadata,
                            &JsonOperationContext::standalone(),
                        )
                        .unwrap()
                        .reserved_bytes()
            })
            .sum::<usize>()
}

#[test]
fn sqlite_baseline_source_snapshot_outlives_the_row_workspace_without_copying_payloads() {
    let (mut doc_file, doc_path) =
        create_temporary_file("sqlite-source-catalog-doc", "jsonl").unwrap();
    let (mut chunk_file, chunk_path) =
        create_temporary_file("sqlite-source-catalog-chunk", "jsonl").unwrap();
    let payload = " ".repeat(600 * 1024);
    for index in 0..64 {
        writeln!(doc_file, "{{\"external_id\":\"d{index}\",\"title\":\"x{payload}\",\"source_uri\":\"file:///source/{index}\",\"metadata\":{{\"payload\":\"{payload}\"}}}}").unwrap();
        writeln!(chunk_file, "{{\"document_external_id\":\"d{index}\",\"chunk_external_id\":\"c{index}\",\"text\":\"the {payload}\",\"ordinal\":0,\"metadata\":{{}}}}").unwrap();
    }
    drop((doc_file, chunk_file, payload));
    let context = JsonOperationContext::standalone();
    let initial = (
        context.available_workspace_bytes(),
        context.available_catalog_bytes(),
        context.available_total_bytes(),
    );
    let mut work = context.catalog_reservation();
    let documents = read_sqlite_baseline_documents(
        doc_path.path(),
        File::open(doc_path.path()).unwrap(),
        &context,
        &mut work,
    )
    .unwrap();
    assert!(work.reserved_bytes() > crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES);
    assert_eq!(context.available_workspace_bytes(), initial.0);
    let (mut catalog, ids) = read_sqlite_baseline_chunks(
        chunk_path.path(),
        File::open(chunk_path.path()).unwrap(),
        &documents,
        &context,
        &mut work,
    )
    .unwrap();
    assert_eq!(
        work.reserved_bytes(),
        baseline_document_bytes(&documents) + baseline_chunk_bytes(&catalog, &ids)
    );
    let boxes = documents
        .values()
        .map(|doc| {
            (
                doc.document_id.clone(),
                &**doc as *const Document,
                doc.title.as_ref().unwrap().as_ptr(),
            )
        })
        .collect::<Vec<_>>();
    install_baseline_documents(&mut catalog, documents, &mut work).unwrap();
    for (id, pointer, title) in boxes {
        let document = &catalog.documents[&id];
        assert_eq!(&**document as *const Document, pointer);
        assert_eq!(document.title.as_ref().unwrap().as_ptr(), title);
    }
    assert_eq!(
        work.reserved_bytes(),
        baseline_document_bytes(&catalog.documents) + baseline_chunk_bytes(&catalog, &ids)
    );
    assert_eq!(context.available_workspace_bytes(), initial.0);
    assert_eq!(
        context.available_catalog_bytes(),
        initial.1 - work.reserved_bytes()
    );
    assert_eq!(
        context.available_total_bytes(),
        initial.2 - work.reserved_bytes()
    );
    let data = SqliteBaselineData {
        catalog,
        chunk_ids_by_external_id: ids,
        json_context: context.clone(),
        _rows_work: work,
    };
    assert_eq!(data.catalog.documents.len(), 64);
    assert_eq!(data.catalog.chunks.len(), 64);
    drop(data);
    assert_eq!(
        (
            context.available_workspace_bytes(),
            context.available_catalog_bytes(),
            context.available_total_bytes()
        ),
        initial
    );
}

#[test]
fn sqlite_baseline_failed_chunk_phase_keeps_the_prior_document_snapshot_owned() {
    let documents_input = concat!(
        "{\"external_id\":\"d1\",\"metadata\":{\"keep\":1}}\n",
        "\n\n\n\n\n\n\n\n",
        "{\"external_id\":\"d2\",\"metadata\":{\"keep\":2}}\n",
    );
    let first = "{\"document_external_id\":\"d1\",\"chunk_external_id\":\"c1\",\"text\":\"the\",\"ordinal\":0,\"metadata\":{}}\n";
    let second = "{\"document_external_id\":\"d2\",\"chunk_external_id\":\"c2\",\"text\":\"the\",\"ordinal\":0,\"metadata\":{}}\n";
    for failure in ["parse", "catalog", "native", "read"] {
        let context = JsonOperationContext::standalone();
        let initial = (
            context.available_workspace_bytes(),
            context.available_catalog_bytes(),
            context.available_total_bytes(),
        );
        let mut busy = context.catalog_reservation();
        let mut retained = context.catalog_reservation();
        let documents = read_sqlite_baseline_documents(
            Path::new("documents.jsonl"),
            documents_input.as_bytes(),
            &context,
            &mut retained,
        )
        .unwrap();
        assert_eq!(documents["d2"].document_id, "doc_10");
        let before = retained.reserved_bytes();
        if failure == "catalog" {
            let (catalog, ids) = read_sqlite_baseline_chunks(
                Path::new("first.jsonl"),
                first.as_bytes(),
                &documents,
                &context,
                &mut retained,
            )
            .unwrap();
            let first_bytes = baseline_chunk_bytes(&catalog, &ids);
            drop((catalog, ids));
            retained.retain(before);
            busy.grow(context.available_catalog_bytes() - first_bytes)
                .unwrap();
        }
        let rows = if failure == "parse" {
            format!("{first}{{invalid second row}}\n")
        } else {
            format!("{first}{second}")
        };
        let path = Path::new("chunks.jsonl");
        let result = match failure {
            "native" => crate::test_alloc::fail_byte_allocation_once(6, || {
                read_sqlite_baseline_chunks(
                    path,
                    rows.as_bytes(),
                    &documents,
                    &context,
                    &mut retained,
                )
            }),
            "read" => {
                let (mut file, temporary) =
                    create_temporary_file("sqlite-source-read-failure", "jsonl").unwrap();
                file.write_all(rows.as_bytes()).unwrap();
                drop(file);
                let (result, stats) = source_probe::measure(
                    Some((1, 1, source_probe::Fault::ReadAfter(first.len()))),
                    || {
                        read_verified_baseline_input(
                            temporary.path(),
                            "chunks",
                            &crate::hash::sha256_hex(rows.as_bytes()),
                            |source| {
                                read_sqlite_baseline_chunks(
                                    path,
                                    source,
                                    &documents,
                                    &context,
                                    &mut retained,
                                )
                            },
                        )
                    },
                );
                assert_eq!(stats.opened, [0, 1, 0]);
                assert_eq!(stats.verified, [0, 0, 0]);
                assert_eq!(stats.injected, 1);
                result
            }
            _ => read_sqlite_baseline_chunks(
                path,
                rows.as_bytes(),
                &documents,
                &context,
                &mut retained,
            ),
        };
        let error = result.err().unwrap();
        assert_eq!(
            error.code(),
            match failure {
                "parse" => ErrorCode::InvalidJsonlSchema,
                "catalog" => ErrorCode::ResourceExhausted,
                _ => ErrorCode::IoError,
            },
            "{failure}: {error}"
        );
        assert_eq!(
            detail_value(&error, "line"),
            Some("2"),
            "{failure}: {error}"
        );
        assert_eq!(detail_value(&error, "path"), Some("chunks.jsonl"));
        if failure == "catalog" {
            assert_eq!(detail_value(&error, "resource"), Some("catalog"));
        }
        assert_eq!(retained.reserved_bytes(), before);
        assert_eq!(
            retained.reserved_bytes(),
            baseline_document_bytes(&documents)
        );
        assert_eq!(
            documents["d1"]
                .metadata
                .get("keep")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(context.available_workspace_bytes(), initial.0);
        drop((documents, retained, busy));
        assert_eq!(
            (
                context.available_workspace_bytes(),
                context.available_catalog_bytes(),
                context.available_total_bytes()
            ),
            initial
        );
    }
}

#[test]
fn sqlite_baseline_document_rekey_covers_both_trees_until_the_old_keys_are_gone() {
    let input = concat!(
        "{\"external_id\":\"first\",\"metadata\":{}}\n",
        "\n\n\n\n\n\n\n\n",
        "{\"external_id\":\"second\",\"metadata\":{}}\n"
    );
    for memory_axis in [false, true] {
        for spare in [None, Some(0), Some(1)] {
            let context = if memory_axis {
                JsonOperationContext::new(std::sync::Arc::new(
                    crate::json::WorkLedger::with_limits(
                        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                        crate::json::WorkLedger::CATALOG_LIMIT_BYTES,
                        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    ),
                ))
            } else {
                JsonOperationContext::standalone()
            };
            let initial = (
                context.available_catalog_bytes(),
                context.available_total_bytes(),
            );
            let mut busy = context.catalog_reservation();
            {
                let mut work = context.catalog_reservation();
                let mut catalog = Catalog::default();
                let documents = read_sqlite_baseline_documents(
                    Path::new("documents.jsonl"),
                    input.as_bytes(),
                    &context,
                    &mut work,
                )
                .unwrap();
                let before = work.reserved_bytes();
                let old_keys = documents.keys().map(String::capacity).sum::<usize>();
                let new_keys = documents
                    .values()
                    .map(|doc| doc.document_id.len())
                    .sum::<usize>();
                let tree = btree_workspace::<String, Box<Document>>(documents.len());
                if let Some(spare) = spare {
                    let available = if memory_axis {
                        context.available_total_bytes()
                    } else {
                        context.available_catalog_bytes()
                    };
                    busy.grow(available - tree - new_keys + spare).unwrap();
                }
                let result = install_baseline_documents(&mut catalog, documents, &mut work);
                if spare == Some(1) {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(
                        detail_value(&error, "resource"),
                        Some(if memory_axis { "memory" } else { "catalog" })
                    );
                    assert!(catalog.documents.is_empty());
                    assert_eq!(work.reserved_bytes(), before);
                } else {
                    result.unwrap();
                    assert_eq!(work.reserved_bytes(), before - old_keys + new_keys);
                    assert_eq!(
                        work.reserved_bytes(),
                        baseline_document_bytes(&catalog.documents)
                    );
                    assert_eq!(catalog.documents.len(), 2);
                    if spare == Some(0) {
                        let available = if memory_axis {
                            context.available_total_bytes()
                        } else {
                            context.available_catalog_bytes()
                        };
                        assert_eq!(available, tree + old_keys);
                    }
                }
            }
            drop(busy);
            assert_eq!(
                (
                    context.available_catalog_bytes(),
                    context.available_total_bytes()
                ),
                initial
            );
        }
    }
    let context = JsonOperationContext::standalone();
    let initial = (
        context.available_catalog_bytes(),
        context.available_total_bytes(),
    );
    let error = {
        let mut work = context.catalog_reservation();
        let mut catalog = Catalog::default();
        let documents = read_sqlite_baseline_documents(
            Path::new("documents.jsonl"),
            input.as_bytes(),
            &context,
            &mut work,
        )
        .unwrap();
        let error = crate::test_alloc::fail_byte_allocation_once(6, || {
            install_baseline_documents(&mut catalog, documents, &mut work)
        })
        .unwrap_err();
        assert_eq!(catalog.documents.len(), 1);
        assert!(work.reserved_bytes() >= baseline_document_bytes(&catalog.documents));
        error
    };
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(
        (
            context.available_catalog_bytes(),
            context.available_total_bytes()
        ),
        initial
    );
}

#[test]
fn sqlite_baseline_transfers_completed_metadata_out_of_the_row_workspace() {
    let (mut documents_file, documents_path) =
        create_temporary_file("sqlite-doc-budget", "jsonl").unwrap();
    let (mut chunks_file, chunks_path) =
        create_temporary_file("sqlite-chunk-budget", "jsonl").unwrap();
    let metadata = format!("{{\"label\":\"{}\"}}", "x".repeat(256 * 1024));
    writeln!(
        documents_file,
        "{{\"external_id\":\"doc\",\"metadata\":{metadata}}}"
    )
    .unwrap();
    writeln!(chunks_file, "{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"chunk\",\"ordinal\":0,\"text\":\"rust\",\"metadata\":{metadata}}}").unwrap();
    drop((documents_file, chunks_file));
    for available in [1024 * 1024, 2 * 1024 * 1024] {
        let context = JsonOperationContext::standalone();
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available,
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
        let mut retained = context.catalog_reservation();
        let documents = read_sqlite_baseline_documents(
            documents_path.path(),
            File::open(documents_path.path()).unwrap(),
            &context,
            &mut retained,
        )
        .unwrap();
        let loaded = read_sqlite_baseline_chunks(
            chunks_path.path(),
            File::open(chunks_path.path()).unwrap(),
            &documents,
            &context,
            &mut retained,
        );
        {
            let (catalog, ids) = loaded.unwrap();
            assert_eq!(catalog.chunks[&ids["chunk"]].text, "rust");
            let expected = documents["doc"]
                .metadata
                .reserve_retained_in(
                    JsonInputKind::DocumentMetadata,
                    &JsonOperationContext::standalone(),
                )
                .unwrap()
                .reserved_bytes()
                + catalog.chunks[&ids["chunk"]]
                    .metadata
                    .reserve_retained_in(
                        JsonInputKind::ChunkMetadata,
                        &JsonOperationContext::standalone(),
                    )
                    .unwrap()
                    .reserved_bytes();
            assert!(expected > 512 * 1024);
            assert_eq!(
                retained.reserved_bytes(),
                baseline_document_bytes(&documents) + baseline_chunk_bytes(&catalog, &ids)
            );
            drop((catalog, ids));
        }
        drop(documents);
        drop(retained);
        busy.grow(available, JsonInputKind::BenchmarkArtifact)
            .unwrap();
    }
}

#[test]
fn sqlite_baseline_metadata_uses_business_depth_roots_and_located_errors() {
    for depth in [128, 129] {
        let nested = format!(
            "{{\"private_marker\":{}0{}}}",
            "[".repeat(depth - 1),
            "]".repeat(depth - 1)
        );
        let (mut file, path) = create_temporary_file("sqlite-metadata-depth", "jsonl").unwrap();
        writeln!(file, "\n{{\"external_id\":\"doc\",\"metadata\":{nested}}}").unwrap();
        drop(file);
        let context = JsonOperationContext::standalone();
        let mut retained = context.catalog_reservation();
        let result = read_sqlite_baseline_documents(
            path.path(),
            File::open(path.path()).unwrap(),
            &context,
            &mut retained,
        );
        if depth == 128 {
            assert_eq!(result.unwrap().len(), 1);
        } else {
            let error = result.err().unwrap();
            assert_eq!(
                detail_value(&error, "json_input_kind"),
                Some("document_metadata")
            );
            assert_eq!(
                detail_value(&error, "json_limit_reason"),
                Some("depth_exceeded")
            );
            assert_eq!(detail_value(&error, "line"), Some("2"));
            assert!(!error.to_string().contains("private_marker"));
        }
        drop(retained);
        let documents = BTreeMap::from([(
            "doc".into(),
            Box::new(Document {
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                created_at: String::new(),
                updated_at: String::new(),
            }),
        )]);
        fs::write(path.path(), format!("\n{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"chunk\",\"ordinal\":0,\"text\":\"rust\",\"metadata\":{nested}}}\n")).unwrap();
        let mut retained = context.catalog_reservation();
        let result = read_sqlite_baseline_chunks(
            path.path(),
            File::open(path.path()).unwrap(),
            &documents,
            &context,
            &mut retained,
        );
        if depth == 128 {
            assert_eq!(result.unwrap().0.chunks.len(), 1);
        } else {
            let error = result.err().unwrap();
            assert_eq!(
                detail_value(&error, "json_input_kind"),
                Some("chunk_metadata")
            );
            assert_eq!(
                detail_value(&error, "json_limit_reason"),
                Some("depth_exceeded")
            );
            assert_eq!(detail_value(&error, "line"), Some("2"));
            assert!(!error.to_string().contains("private_marker"));
        }
        drop(retained);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
    }
}

#[test]
fn sqlite_baseline_single_text_copy_preserves_fts_ranking_and_metadata_filters() {
    let chunk_id = |index| format!("c-{index:03}'\t\n\0\u{4e2d}");
    let (mut doc_file, doc_path) = create_temporary_file("sqlite-doc-rank", "jsonl").unwrap();
    let (mut chunk_file, chunk_path) = create_temporary_file("sqlite-chunk-rank", "jsonl").unwrap();
    writeln!(doc_file, "{{\"external_id\":\"doc\",\"source_uri\":\"file:///source\",\"metadata\":{{\"kind\":\"keep\",\"source_uri\":\"file:///metadata\"}}}}").unwrap();
    for index in 0..100 {
        let id = JsonValue::String(chunk_id(index)).to_compact_string();
        writeln!(chunk_file, "{{\"document_external_id\":\"doc\",\"chunk_external_id\":{id},\"ordinal\":{index},\"text\":\"rust text with ' quote\",\"metadata\":{{\"rank\":{index}}}}}").unwrap();
    }
    drop((doc_file, chunk_file));
    let context = JsonOperationContext::standalone();
    let mut retained = context.catalog_reservation();
    let documents = read_sqlite_baseline_documents(
        doc_path.path(),
        File::open(doc_path.path()).unwrap(),
        &context,
        &mut retained,
    )
    .unwrap();
    let (mut catalog, ids) = read_sqlite_baseline_chunks(
        chunk_path.path(),
        File::open(chunk_path.path()).unwrap(),
        &documents,
        &context,
        &mut retained,
    )
    .unwrap();
    install_baseline_documents(&mut catalog, documents, &mut retained).unwrap();
    let data = SqliteBaselineData {
        catalog,
        chunk_ids_by_external_id: ids,
        json_context: context.clone(),
        _rows_work: retained,
    };
    let query = BenchmarkQuery {
        query_id: "q".into(),
        text: Some("rust".into()),
        filter: parse(
            r#"{"doc.kind":"keep","doc.source_uri":"file:///metadata","chunk.rank":{"$gte":50}}"#,
        )
        .unwrap(),
        top_k: 100,
        embedding: None,
        relevant_chunk_ids: BTreeSet::new(),
    };
    let filter = FilterExpr::parse(query.filter.clone()).unwrap();
    for index in 0..100 {
        assert_eq!(
            sqlite_baseline_candidate_matches_filter(&chunk_id(index), &filter, &data),
            index >= 50
        );
    }
    if sqlite3_fts5_available() {
        let (file, path) = create_temporary_file("sqlite-single-copy", "sqlite").unwrap();
        drop(file);
        build_sqlite_fts5_database(path.path(), &data.catalog).unwrap();
        let hits = sqlite_fts5_hits(path.path(), &query, &data, 100).unwrap();
        assert_eq!(
            hits.iter()
                .map(|hit| hit.chunk_id.clone())
                .collect::<Vec<_>>(),
            (50..100).map(chunk_id).collect::<Vec<_>>()
        );
        assert!(hits
            .iter()
            .all(|hit| hit.score.to_bits() == 0.000001f64.to_bits()));
    }
    drop(data);
    context
        .reservation()
        .grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
}

#[test]
fn sqlite_baseline_token_hint_failure_stops_at_its_row_and_allows_clean_retry() {
    let documents = BTreeMap::from([(
        "doc".into(),
        Box::new(Document {
            document_id: "doc_1".into(),
            collection_id: "col_1".into(),
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            created_at: String::new(),
            updated_at: String::new(),
        }),
    )]);
    let part_bytes = 131_071;
    let word = "P".repeat(part_bytes);
    let first = "{\"document_external_id\":\"doc\",\"chunk_external_id\":\"first\",\"ordinal\":0,\"text\":\"the and\",\"metadata\":{}}\n";
    let second = format!("{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"second\",\"ordinal\":1,\"text\":\"{word}_{word}\",\"metadata\":{{}}}}\n");
    let rows = format!("{first}{second}{{invalid third row}}\n");
    let path = Path::new("token-hints.jsonl");
    let context = JsonOperationContext::standalone();
    let capacity = context.available_workspace_bytes();
    let mut retained = context.catalog_reservation();
    let error = crate::test_alloc::fail_byte_allocation_once(part_bytes, || {
        read_sqlite_baseline_chunks(path, rows.as_bytes(), &documents, &context, &mut retained)
    })
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::IoError, "{error}");
    assert_eq!(detail_value(&error, "line"), Some("2"));
    assert_eq!(detail_value(&error, "path"), Some("token-hints.jsonl"));
    drop(retained);
    assert_eq!(context.available_workspace_bytes(), capacity);
    let rows = format!("{first}{second}");
    let mut retained = context.catalog_reservation();
    let (catalog, ids) =
        read_sqlite_baseline_chunks(path, rows.as_bytes(), &documents, &context, &mut retained)
            .unwrap();
    assert_eq!(catalog.chunks[&ids["first"]].token_count, 0);
    assert_eq!(catalog.chunks[&ids["second"]].token_count, 2);
    assert_eq!(catalog.chunks.len(), 2);
    drop((catalog, ids));
    drop(retained);
    assert_eq!(context.available_workspace_bytes(), capacity);
}

#[test]
fn sqlite_baseline_config_reports_the_configured_executable() {
    let dir = text_only_benchmark_dataset("sqlite-executable-config");
    let dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
    let configs = sqlite_baseline_configs_in(&dataset, false, &JsonBuilder::host()).unwrap();
    assert_eq!(
        configs.as_array().unwrap()[0]
            .get("config")
            .unwrap()
            .get("sqlite_cli")
            .and_then(JsonValue::as_str),
        sqlite_cli().to_str(),
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sqlite_executable_identity_is_lossless_and_rejects_invalid_labels() {
    use std::ffi::OsStr;
    for seed in 0..100 {
        let executable = format!("/tools/sqlite {seed}'/\u{4e2d}/sqlite3");
        assert_eq!(
            sqlite_cli_label(OsStr::new(&executable)).unwrap(),
            executable
        );
    }
    assert_eq!(sqlite_cli_label(OsStr::new("sqlite3")).unwrap(), "sqlite3");
    for invalid in ["", "  ", "sqlite\0"] {
        assert_eq!(
            sqlite_cli_label(OsStr::new(invalid)).unwrap_err().code(),
            ErrorCode::CliUsage
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(
            sqlite_cli_label(OsStr::from_bytes(b"sqlite\xff"))
                .unwrap_err()
                .code(),
            ErrorCode::CliUsage
        );
    }
}

#[test]
fn sqlite_vector_producer_streams_beyond_file_and_cumulative_vector_work_limits() {
    let dir = text_only_benchmark_dataset("sqlite-stream-large-vectors");
    let mut dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
    let path = dir.join("embeddings.jsonl");
    let mut source = std::io::BufWriter::new(File::create(&path).unwrap());
    let vector = format!("[{}]", vec!["0.125"; 768].join(","));
    for row in 0..16_384 {
        writeln!(source, "{{\"chunk_external_id\":\"chunk-{row}\",\"model_name\":\"m\",\"dim\":768,\"vector\":{vector}}}").unwrap();
    }
    source.flush().unwrap();
    drop(source);
    assert!(fs::metadata(&path).unwrap().len() > crate::import::MAX_JSONL_BYTES);
    assert!(
        16_384 * 768 * std::mem::size_of::<f32>()
            > crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    dataset.manifest.embeddings_sha256 =
        Some(crate::hash::sha256_reader(File::open(&path).unwrap()).unwrap());
    dataset.embeddings_path = Some(path);
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    let mut output = SqliteVectorSink::new(&context);
    write_sqlite_vectors(&mut output, &dataset, &context).unwrap();
    assert_eq!(output.inserts, 16_384);
    assert!(output.committed);
    assert!(output.sampled_peak_work < 1024 * 1024);
    assert_eq!(context.available_workspace_bytes(), available);
    fs::remove_dir_all(dir).unwrap();
}

struct SqliteVectorSink<'a> {
    context: &'a JsonOperationContext,
    available: usize,
    inserts: usize,
    committed: bool,
    sampled_peak_work: usize,
    fail_insert: Option<usize>,
}

impl<'a> SqliteVectorSink<'a> {
    fn new(context: &'a JsonOperationContext) -> Self {
        Self {
            context,
            available: context.available_workspace_bytes(),
            inserts: 0,
            committed: false,
            sampled_peak_work: 0,
            fail_insert: None,
        }
    }
}

impl Write for SqliteVectorSink<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes == b"INSERT OR REPLACE INTO chunks_vec VALUES(" {
            if self.fail_insert == Some(self.inserts + 1) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "injected SQL input failure",
                ));
            }
            self.inserts += 1;
            self.sampled_peak_work = self
                .sampled_peak_work
                .max(self.available - self.context.available_workspace_bytes());
        }
        if bytes == b"COMMIT;\n" {
            self.committed = true;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn sqlite_vector_producer_verifies_its_stream_before_commit_and_refunds_failures() {
    let dir = text_only_benchmark_dataset("sqlite-stream-vector-errors");
    let mut dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
    let input = concat!(
        "{\"chunk_external_id\":\"first\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}\n",
        "{\"chunk_external_id\":\"second\",\"model_name\":\"m\",\"dim\":2,\"vector\":[0,1]}\n"
    );
    let path = dir.join("embeddings.jsonl");
    dataset.embeddings_path = Some(path.clone());
    dataset.manifest.embeddings_sha256 = Some(crate::hash::sha256_hex(input.as_bytes()));
    for fault in 0..3 {
        fs::write(
            &path,
            if fault == 1 {
                format!("{input} ")
            } else {
                input.into()
            },
        )
        .unwrap();
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut output = SqliteVectorSink::new(&context);
        if fault == 2 {
            output.fail_insert = Some(2);
        }
        let result = write_sqlite_vectors(&mut output, &dataset, &context);
        if fault == 0 {
            result.unwrap();
            assert_eq!(output.inserts, 2);
            assert!(output.committed);
        } else {
            let error = result.unwrap_err();
            assert_eq!(
                error.code(),
                if fault == 1 {
                    ErrorCode::ChecksumMismatch
                } else {
                    ErrorCode::IoError
                }
            );
            assert!(!output.committed);
            assert_eq!(detail_value(&error, "path"), path.to_str());
            if fault == 1 {
                assert_eq!(output.inserts, 2);
                assert_eq!(detail_value(&error, "artifact_kind"), Some("embeddings"));
            } else {
                assert_eq!(output.inserts, 1);
                assert_eq!(detail_value(&error, "line"), Some("2"));
            }
        }
        assert_eq!(context.available_workspace_bytes(), available);
    }
    fs::write(&path, input).unwrap();
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    let mut output = SqliteVectorSink::new(&context);
    let (result, scans) = source_probe::measure(
        Some((
            2,
            1,
            source_probe::Fault::ReadAfter(input.find('\n').unwrap() + 1),
        )),
        || write_sqlite_vectors(&mut output, &dataset, &context),
    );
    assert_eq!(result.unwrap_err().code(), ErrorCode::IoError);
    assert_eq!(scans.opened, [0, 0, 1]);
    assert_eq!(scans.verified, [0, 0, 0]);
    assert_eq!(scans.injected, 1);
    assert_eq!(output.inserts, 1);
    assert!(!output.committed);
    assert_eq!(context.available_workspace_bytes(), available);
    let mut output = SqliteVectorSink::new(&context);
    let (result, scans) = source_probe::measure(None, || {
        write_sqlite_vectors(&mut output, &dataset, &context)
    });
    result.unwrap();
    assert_eq!(scans.opened, [0, 0, 1]);
    assert_eq!(scans.verified, scans.opened);
    assert_eq!(scans.bytes, [0, 0, input.len()]);
    assert!(output.committed);
    assert_eq!(context.available_workspace_bytes(), available);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sqlite_exact_baseline_stats_count_only_prefiltered_distances() {
    let mut catalog = Catalog::default();
    let keep = Chunk {
        chunk_id: "c_keep".into(),
        document_id: "d".into(),
        collection_id: "col".into(),
        chunk_external_id: "keep".into(),
        text: "keep".into(),
        text_hash: "h1".into(),
        ordinal: 0,
        token_count: 1,
        metadata: parse(r#"{"group":"keep"}"#).unwrap(),
        embedding_id: None,
        source_span: None,
        created_at: "1970-01-01T00:00:00Z".into(),
    };
    let drop = Chunk {
        chunk_id: "c_drop".into(),
        document_id: "d".into(),
        collection_id: "col".into(),
        chunk_external_id: "drop".into(),
        text: "drop".into(),
        text_hash: "h2".into(),
        ordinal: 1,
        token_count: 1,
        metadata: parse(r#"{"group":"drop"}"#).unwrap(),
        embedding_id: None,
        source_span: None,
        created_at: "1970-01-01T00:00:00Z".into(),
    };
    for (id, external_id) in [
        ("c_wrong", "wrong"),
        ("c_missing", "missing"),
        ("c_tie", "t'ie"),
    ] {
        let mut chunk = keep.clone();
        chunk.chunk_id = id.into();
        chunk.chunk_external_id = external_id.into();
        catalog.chunks.insert(id.into(), Box::new(chunk));
    }
    catalog.chunks.insert(keep.chunk_id.clone(), Box::new(keep));
    catalog.chunks.insert(drop.chunk_id.clone(), Box::new(drop));
    let json_context = JsonOperationContext::standalone();
    let baseline_data = SqliteBaselineData {
        catalog,
        chunk_ids_by_external_id: BTreeMap::from([
            ("keep".to_owned(), "c_keep".to_owned()),
            ("drop".to_owned(), "c_drop".to_owned()),
            ("wrong".to_owned(), "c_wrong".to_owned()),
            ("missing".to_owned(), "c_missing".to_owned()),
            ("t'ie".to_owned(), "c_tie".to_owned()),
        ]),
        _rows_work: json_context.catalog_reservation(),
        json_context,
    };
    let mut query = BenchmarkQuery {
        query_id: "q".into(),
        text: None,
        filter: parse(r#"{"chunk.group":"keep"}"#).unwrap(),
        top_k: 10,
        embedding: Some(QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        }),
        relevant_chunk_ids: BTreeSet::new(),
    };

    if !sqlite3_sqlite_vec_available() {
        return;
    }
    let db = TempFileGuard::new(std::env::temp_dir().join(format!(
        "cofferdb-vector-test-{}.sqlite",
        std::process::id()
    )));
    let dir = text_only_benchmark_dataset("sqlite-vector-prefilter");
    let mut dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
    let input = concat!(
        "{\"chunk_external_id\":\"keep\",\"model_name\":\"m\",\"dim\":2,\"vector\":[0,1]}\n",
        "{\"chunk_external_id\":\"drop\",\"model_name\":\"m\",\"dim\":2,\"vector\":[0,1]}\n",
        "{\"chunk_external_id\":\"wrong\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0,0]}\n",
        "{\"chunk_external_id\":\"t'ie\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}\n",
        "{\"chunk_external_id\":\"keep\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}\n"
    );
    let embeddings = dir.join("embeddings.jsonl");
    fs::write(&embeddings, input).unwrap();
    dataset.embeddings_path = Some(embeddings);
    dataset.manifest.embeddings_sha256 = Some(crate::hash::sha256_hex(input.as_bytes()));
    build_sqlite_vector_database(db.path(), &dataset, &baseline_data.json_context).unwrap();
    let (hits, stats) = sqlite_vec_exact_hits(db.path(), &query, &baseline_data, 10).unwrap();

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].chunk_id, "keep");
    assert_eq!(hits[1].chunk_id, "t'ie");
    assert_eq!(hits[0].raw_score.to_bits(), 1.0f64.to_bits());
    assert_eq!(hits[0].normalized_score.to_bits(), 1.0f64.to_bits());
    assert_eq!(hits[1].raw_score.to_bits(), hits[0].raw_score.to_bits());
    assert_eq!(
        hits[1].normalized_score.to_bits(),
        hits[0].normalized_score.to_bits()
    );
    assert_eq!(stats.vector_scan_count, 2);
    assert_eq!(stats.distance_ops, 4);
    assert_eq!(stats.fusion_candidate_count, 2);
    query.filter = JsonValue::object();
    let (all, stats) = sqlite_vec_exact_hits(db.path(), &query, &baseline_data, 10).unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].chunk_id, "keep");
    assert_eq!(all[1].chunk_id, "t'ie");
    assert_eq!(all[2].chunk_id, "drop");
    assert_eq!(all[2].raw_score.to_bits(), 0.0f64.to_bits());
    assert_eq!(all[2].normalized_score.to_bits(), 0.5f64.to_bits());
    assert_eq!(stats.vector_scan_count, 3);
    assert_eq!(stats.distance_ops, 6);
    let (limited, stats) = sqlite_vec_exact_hits(db.path(), &query, &baseline_data, 1).unwrap();
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].chunk_id, "keep");
    assert_eq!(stats.vector_scan_count, 3);
    assert_eq!(stats.distance_ops, 6);
    assert_eq!(stats.fusion_candidate_count, 1);
    query.filter = parse(r#"{"chunk.group":"absent"}"#).unwrap();
    let (none, stats) = sqlite_vec_exact_hits(db.path(), &query, &baseline_data, 10).unwrap();
    assert_eq!(none.len(), 0);
    assert_eq!(stats.vector_scan_count, 0);
    assert_eq!(stats.distance_ops, 0);
    query.filter = parse(r#"{"chunk.group":"keep"}"#).unwrap();
    query.embedding = Some(QueryEmbeddingInput {
        model_name: Some("m".into()),
        dim: 3,
        vector: vec![1.0, 0.0, 0.0],
    });
    let (three_dimensional, stats) =
        sqlite_vec_exact_hits(db.path(), &query, &baseline_data, 10).unwrap();
    assert_eq!(three_dimensional.len(), 1);
    assert_eq!(three_dimensional[0].chunk_id, "wrong");
    assert_eq!(three_dimensional[0].raw_score.to_bits(), 1.0f64.to_bits());
    assert_eq!(stats.vector_scan_count, 1);
    assert_eq!(stats.distance_ops, 3);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn concurrent_temporary_files_are_unique_private_and_cleaned_on_error() {
    let workers = (0..16)
        .map(|_| {
            std::thread::spawn(|| {
                let mut paths = Vec::new();
                for _ in 0..8 {
                    let (mut file, guard) =
                        create_temporary_file("concurrent-test", "tmp").unwrap();
                    let path = guard.path().to_path_buf();
                    file.write_all(b"private temporary contents").unwrap();
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
                    }
                    assert!(path.exists());
                    drop(file);
                    drop(guard);
                    assert!(!path.exists());
                    paths.push(path);
                }
                paths
            })
        })
        .collect::<Vec<_>>();
    let paths = workers
        .into_iter()
        .flat_map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(paths.iter().collect::<BTreeSet<_>>().len(), 128);
    let mut path = PathBuf::new();
    let result: Result<()> = (|| {
        let (_file, guard) = create_temporary_file("error-test", "tmp")?;
        path = guard.path().to_path_buf();
        Err(Error::new(ErrorCode::IoError, "injected operation failure"))
    })();
    assert!(result.is_err());
    assert!(!path.exists());
}

#[test]
fn sqlite_baseline_temp_guard_removes_file_on_drop() {
    let path = std::env::temp_dir().join(format!(
        "cofferdb-sqlite-baseline-guard-{}-{}.sqlite",
        std::process::id(),
        crate::time::now_rfc3339().replace(':', "-")
    ));
    std::fs::write(&path, b"temporary baseline db").unwrap();
    {
        let _guard = TempFileGuard::new(path.clone());
        assert!(path.exists());
    }
    assert!(!path.exists());
}

#[test]
fn benchmark_empty_vector_results_report_prefiltered_query_level_stats() {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "cofferdb-cli-benchmark-empty-vector-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("benchmark-empty-vector.coffer");
    std::fs::write(
        dir.join("documents.jsonl"),
        r#"{"external_id":"doc","metadata":{}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("chunks.jsonl"),
        r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"semantic wal recovery","metadata":{"lang":"rust"}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("embeddings.jsonl"),
        r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("queries.jsonl"),
        r#"{"query_id":"q1","text":"semantic wal","filter":{"chunk.lang":"python"},"top_k":10,"embedding":[1.0,0.0]}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("qrels.jsonl"),
        r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
    )
    .unwrap();
    let documents_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("documents.jsonl")).unwrap());
    let chunks_sha256 = crate::hash::sha256_hex(&std::fs::read(dir.join("chunks.jsonl")).unwrap());
    let embeddings_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("embeddings.jsonl")).unwrap());
    let queries_sha256 =
        crate::hash::sha256_hex(&std::fs::read(dir.join("queries.jsonl")).unwrap());
    let qrels_sha256 = crate::hash::sha256_hex(&std::fs::read(dir.join("qrels.jsonl")).unwrap());
    std::fs::write(
        dir.join("dataset_manifest.json"),
        format!(
            concat!(
                r#"{{"schema_version":"0.1","dataset_id":"bench-empty-vector","version":"0.1","#,
                r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                r#""qrels":"qrels.jsonl","query_embedding_source":"inline","#,
                r#""embedding_model":"m","query_embedding_model":"m","#,
                r#""query_embedding_dim":2,"hashes":{{"#,
                r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                r#""qrels_sha256":"{}"}}"#,
                r#"}}"#
            ),
            documents_sha256, chunks_sha256, embeddings_sha256, queries_sha256, qrels_sha256
        ),
    )
    .unwrap();

    let output = run(vec![
        "coffer".into(),
        "benchmark".into(),
        db_path.display().to_string(),
        "--dataset".into(),
        dir.display().to_string(),
    ])
    .unwrap();
    let json = parse(&output).unwrap();
    let result = json.get("result").and_then(JsonValue::as_object).unwrap();
    let metrics = result
        .get("metrics")
        .and_then(JsonValue::as_object)
        .unwrap();
    assert_eq!(
        metrics.get("vector_scan_count").and_then(JsonValue::as_f64),
        Some(1.0)
    );
    assert_eq!(
        metrics.get("distance_ops").and_then(JsonValue::as_f64),
        Some(0.0)
    );
    let runs = result.get("runs").and_then(JsonValue::as_array).unwrap();
    let hybrid_warm = runs
        .iter()
        .find(|run| {
            run.get("query_mode").and_then(JsonValue::as_str) == Some("hybrid")
                && run.get("cache_state").and_then(JsonValue::as_str) == Some("warm")
        })
        .unwrap();
    assert_eq!(
        hybrid_warm
            .get("metrics")
            .and_then(|stats| stats.get("vector_scan_count"))
            .and_then(JsonValue::as_f64),
        Some(1.0)
    );
    let per_query = hybrid_warm
        .get("per_query")
        .and_then(JsonValue::as_array)
        .unwrap();
    assert!(per_query[0]
        .get("ranked_chunk_ids")
        .and_then(JsonValue::as_array)
        .unwrap()
        .is_empty());
    assert_eq!(
        per_query[0]
            .get("execution_stats")
            .and_then(|stats| stats.get("vector_scan_count"))
            .and_then(JsonValue::as_f64),
        Some(1.0)
    );
    assert_eq!(
        per_query[0]
            .get("execution_stats")
            .and_then(|stats| stats.get("distance_ops"))
            .and_then(JsonValue::as_f64),
        Some(0.0)
    );
    assert_eq!(
        per_query[0]
            .get("query_context")
            .and_then(|context| context.get("execution_stats"))
            .and_then(|stats| stats.get("vector_scan_count"))
            .and_then(JsonValue::as_f64),
        Some(1.0)
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn benchmark_hardware_json_records_cpu_and_os() {
    let hardware = benchmark_hardware_json();
    assert!(hardware
        .get("cpu")
        .and_then(JsonValue::as_str)
        .map(|cpu| cpu.contains("core") || cpu.contains("CPU"))
        .unwrap_or(false));
    assert!(hardware
        .get("os")
        .and_then(JsonValue::as_str)
        .map(|os| !os.trim().is_empty())
        .unwrap_or(false));
    assert_eq!(
        hardware.get("storage").and_then(JsonValue::as_str),
        Some("local-storage")
    );
    assert!(hardware.get("cpu_model").is_none());
    assert!(hardware.get("cpu_cores").is_none());
    assert!(hardware.get("storage_type").is_none());
    assert!(hardware.get("os_name").is_none());
    assert!(hardware.get("os_version").is_none());
}

#[test]
fn benchmark_result_contract_rejects_none_vector_modes() {
    let mut invalid_run = object_from_pairs([
        ("query_mode", JsonValue::String("hybrid".into())),
        ("cache_state", JsonValue::String("warm".into())),
    ]);
    let result = benchmark_contract_result(
        "none",
        vec![invalid_run.clone()],
        vec![object_from_pairs([(
            "requires_query_embedding",
            JsonValue::Bool(false),
        )])],
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].query_mode")
    );

    let mut baseline_run = benchmark_contract_baseline_run("warm");
    insert_object_field(
        &mut baseline_run,
        "query_mode",
        JsonValue::String("hybrid".into()),
    );
    let per_query = baseline_run
        .as_object_mut()
        .unwrap()
        .get_mut("per_query")
        .unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract baseline run must contain per_query array");
    };
    insert_object_field(
        &mut items[0],
        "score_kind",
        JsonValue::String("sqlite_fts5_vec_weighted_sum".into()),
    );
    let result = benchmark_contract_completed_baseline_result(vec![baseline_run]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].query_mode")
    );

    insert_object_field(
        &mut invalid_run,
        "query_mode",
        JsonValue::String("text_bm25".into()),
    );
    let result = benchmark_contract_result(
        "none",
        vec![invalid_run],
        vec![object_from_pairs([(
            "requires_query_embedding",
            JsonValue::Bool(true),
        )])],
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].requires_query_embedding")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("metrics").unwrap(),
        "vector_scan_count",
        JsonValue::Number(1.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("metrics.vector_scan_count")
    );

    let mut run = benchmark_contract_run("text_bm25");
    run.as_object_mut()
        .unwrap()
        .get_mut("metrics")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("distance_ops".into(), JsonValue::Number(1.0));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].metrics.distance_ops")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("execution_stats")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("vector_scan_count".into(), JsonValue::Number(1.0));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].execution_stats.vector_scan_count")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .and_then(|context| context.get_mut("execution_stats"))
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("distance_ops".into(), JsonValue::Number(1.0));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.execution_stats.distance_ops")
    );
}

#[test]
fn benchmark_result_contract_requires_public_fields() {
    let result = object_from_pairs([
        ("schema_version", JsonValue::String("0.1".into())),
        ("cofferdb_version", JsonValue::String("0.1.0".into())),
    ]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("git_commit")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        &mut result,
        "dataset_manifest_hash",
        JsonValue::String("typo".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset_manifest_hash")
    );
}

#[test]
fn benchmark_result_contract_validates_index_mode_and_dataset_identity() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        &mut result,
        "index_build_mode",
        JsonValue::String("latest_batch_only".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("index_build_mode")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("dataset")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .remove("queries_sha256");
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.queries_sha256")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("dataset")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("dataset_id".into(), JsonValue::String("other".into()));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset_id")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("dataset")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("documents_sha256".into(), JsonValue::String(String::new()));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.documents_sha256")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("dataset")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("documents_hash".into(), JsonValue::String("typo".into()));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.documents_hash")
    );
}

#[test]
fn benchmark_result_contract_validates_profile_and_durability_enums() {
    let mut healthy = benchmark_contract_completed_baseline_result(vec![
        benchmark_contract_baseline_run("cold"),
        benchmark_contract_baseline_run("warm"),
    ]);
    insert_object_field(
        &mut healthy,
        "runs",
        JsonValue::Array(vec![
            benchmark_contract_run_with_cache("text_bm25", "cold"),
            benchmark_contract_run_with_cache("text_bm25", "warm"),
        ]),
    );
    validate_benchmark_result_contract(&healthy).unwrap();

    for (profile, candidates, cache) in [
        ("expanded", 500.0, 512.0),
        ("balanced", 200.0, 256.0),
        ("constrained", 50.0, 64.0),
    ] {
        let mut valid = healthy.clone();
        insert_object_field(&mut valid, "profile", JsonValue::String(profile.into()));
        let params = valid
            .as_object_mut()
            .unwrap()
            .get_mut("profile_params")
            .unwrap();
        for field in ["bm25_candidates", "vector_candidates", "fusion_candidates"] {
            insert_object_field(params, field, JsonValue::Number(candidates));
        }
        insert_object_field(params, "cache_budget_mb", JsonValue::Number(cache));
        let counts = valid
            .as_object_mut()
            .unwrap()
            .get_mut("candidate_counts")
            .unwrap();
        for field in ["bm25", "vector", "fusion"] {
            insert_object_field(counts, field, JsonValue::Number(candidates));
        }
        for field in ["runs", "baseline_results"] {
            let JsonValue::Array(runs) = valid.as_object_mut().unwrap().get_mut(field).unwrap()
            else {
                panic!("runs array");
            };
            for run in runs {
                let JsonValue::Array(rows) =
                    run.as_object_mut().unwrap().get_mut("per_query").unwrap()
                else {
                    panic!("per_query array");
                };
                for row in rows {
                    let context = row
                        .as_object_mut()
                        .unwrap()
                        .get_mut("query_context")
                        .unwrap();
                    insert_object_field(context, "profile", JsonValue::String(profile.into()));
                    let limits = context
                        .as_object_mut()
                        .unwrap()
                        .get_mut("candidate_limits")
                        .unwrap();
                    for branch in ["bm25", "vector", "fusion"] {
                        insert_object_field(limits, branch, JsonValue::Number(candidates));
                    }
                }
            }
        }
        let before = valid.to_compact_string();
        validate_benchmark_result_contract(&valid).unwrap();
        assert_eq!(valid.to_compact_string(), before);
    }

    for value in [
        "quality-first",
        "quality_first",
        "resource-first",
        "resource_first",
        "performance",
        "eco",
        "QualityFirst",
        "ResourceFirst",
        "Expanded",
        "Balanced",
        "Constrained",
        "EXPANDED",
        "BALANCED",
        "CONSTRAINED",
        " expanded",
        "constrained ",
        "expanded-budget",
        "",
        "help",
        "balanced-v2",
    ]
    .into_iter()
    .map(|name| JsonValue::String(name.into()))
    .chain([
        JsonValue::Null,
        JsonValue::Number(1.0),
        JsonValue::Bool(false),
        JsonValue::Array(Vec::new()),
        JsonValue::object(),
    ]) {
        // Result validation is an internal call, not a public JSON readback command.
        // Mutate exactly one profile slot in an otherwise valid complete result.
        for (field, run_index) in [
            ("profile", 0),
            ("runs", 0),
            ("runs", 1),
            ("baseline_results", 0),
            ("baseline_results", 1),
        ] {
            let mut result = healthy.clone();
            let expected_field = if field == "profile" {
                insert_object_field(&mut result, "profile", value.clone());
                "profile".to_owned()
            } else {
                let JsonValue::Array(runs) =
                    result.as_object_mut().unwrap().get_mut(field).unwrap()
                else {
                    panic!("runs array");
                };
                let JsonValue::Array(rows) = runs[run_index]
                    .as_object_mut()
                    .unwrap()
                    .get_mut("per_query")
                    .unwrap()
                else {
                    panic!("per_query array");
                };
                let context = rows[0]
                    .as_object_mut()
                    .unwrap()
                    .get_mut("query_context")
                    .unwrap();
                insert_object_field(context, "profile", value.clone());
                format!("{field}[].per_query[].query_context.profile")
            };
            let before = result.to_compact_string();
            let err = validate_benchmark_result_contract(&result).unwrap_err();
            assert_eq!(
                err.code(),
                ErrorCode::InvalidJsonlSchema,
                "{field}/{run_index}/{value:?}"
            );
            assert_eq!(detail_value(&err, "field"), Some(expected_field.as_str()));
            assert_eq!(result.to_compact_string(), before);
        }
    }

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(&mut result, "durability", JsonValue::String("sync".into()));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(detail_value(&err, "field"), Some("durability"));
}

#[test]
fn benchmark_result_contract_validates_query_mode_enum() {
    let run = benchmark_contract_run("semantic_magic");
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].query_mode")
    );

    let run = benchmark_contract_run("text_bm25");
    let mut baseline = benchmark_contract_baseline_run("warm");
    insert_object_field(
        &mut baseline,
        "query_mode",
        JsonValue::String("sqlite-custom".into()),
    );
    let mut result = benchmark_contract_result(
        "none",
        vec![run],
        vec![benchmark_contract_baseline_config()],
    );
    insert_object_field(
        &mut result,
        "baseline_results",
        JsonValue::Array(vec![baseline]),
    );
    insert_object_field(
        &mut result,
        "baseline_status",
        JsonValue::String("completed".into()),
    );
    insert_object_field(&mut result, "baseline_status_reason", JsonValue::Null);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].query_mode")
    );
}

#[test]
fn benchmark_result_contract_validates_internal_run_engine() {
    let mut run = benchmark_contract_run("text_bm25");
    insert_object_field(&mut run, "engine", JsonValue::String("sqlite".into()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].engine")
    );

    let mut run = benchmark_contract_run("text_bm25");
    insert_object_field(&mut run, "engine", JsonValue::String(String::new()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].engine")
    );
}

#[test]
fn benchmark_result_contract_validates_score_kind_matches_mode() {
    let mut run = benchmark_contract_run("hybrid");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    insert_object_field(
        &mut items[0],
        "score_kind",
        JsonValue::String("cofferdb_bm25_final_score".into()),
    );
    let result = benchmark_contract_result("sidecar", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].score_kind")
    );

    let mut baseline = benchmark_contract_baseline_run("warm");
    let per_query = baseline
        .as_object_mut()
        .unwrap()
        .get_mut("per_query")
        .unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract baseline run must contain per_query array");
    };
    insert_object_field(
        &mut items[0],
        "score_kind",
        JsonValue::String("cofferdb_bm25_final_score".into()),
    );
    let result = benchmark_contract_completed_baseline_result(vec![baseline]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].per_query[].score_kind")
    );
}

#[test]
fn benchmark_result_contract_validates_version_fields() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        &mut result,
        "schema_version",
        JsonValue::String("0.2".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("schema_version")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(&mut result, "git_commit", JsonValue::String("".into()));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("git_commit")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(&mut result, "file_format_version", JsonValue::Number(999.0));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("file_format_version")
    );
}

#[test]
fn benchmark_result_contract_validates_query_embedding_identity() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("dataset").unwrap(),
        "query_embedding_model",
        JsonValue::String("model-should-be-null".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.query_embedding_model")
    );

    let mut result = benchmark_contract_result("sidecar", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("dataset").unwrap(),
        "query_embeddings_sha256",
        JsonValue::Null,
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.query_embeddings_sha256")
    );

    let mut result = benchmark_contract_result("inline", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("dataset").unwrap(),
        "query_embeddings_sha256",
        JsonValue::String("inline-should-not-have-sidecar-hash".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.query_embeddings_sha256")
    );
}

#[test]
fn benchmark_result_contract_validates_dataset_embedding_identity() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("dataset").unwrap(),
        "embedding_model",
        JsonValue::String("model-should-be-null".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.embedding_model")
    );

    let mut result = benchmark_contract_result("sidecar", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("dataset").unwrap(),
        "embeddings_sha256",
        JsonValue::Null,
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.embeddings_sha256")
    );

    let mut result = benchmark_contract_result("inline", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("dataset").unwrap(),
        "embedding_dim",
        JsonValue::Number(4.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.embedding_dim")
    );
}

#[test]
fn benchmark_result_contract_validates_dataset_chunking() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("dataset").unwrap(),
        "chunking",
        object_from_pairs([
            ("chunk_size", JsonValue::Number(800.0)),
            ("chunk_overlap", JsonValue::Number(120.0)),
        ]),
    );
    validate_benchmark_result_contract(&result).unwrap();

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("dataset").unwrap(),
        "chunking",
        object_from_pairs([
            ("chunk_size", JsonValue::Number(800.0)),
            ("chunk_overlap", JsonValue::Number(800.0)),
        ]),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.chunking.chunk_overlap")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("dataset")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .remove("chunking");
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("dataset.chunking")
    );
}

#[test]
fn benchmark_result_contract_validates_hardware_schema() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("hardware")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .remove("os");
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("hardware.os")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("hardware").unwrap(),
        "cpu",
        JsonValue::String("test CPU".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("hardware.cpu")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("hardware").unwrap(),
        "cpu_model",
        JsonValue::String("test CPU".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("hardware.cpu_model")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("hardware").unwrap(),
        "memory_gb",
        JsonValue::Number(0.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("hardware.memory_gb")
    );
}

#[test]
fn benchmark_result_contract_validates_warning_schema() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(&mut result, "warnings", JsonValue::String("warn".into()));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("warnings")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        &mut result,
        "warnings",
        JsonValue::Array(vec![object_from_pairs([
            ("message", JsonValue::String("baseline missing".into())),
            ("reason", JsonValue::Null),
        ])]),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("warnings[].code")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        &mut result,
        "warnings",
        JsonValue::Array(vec![object_from_pairs([
            (
                "code",
                JsonValue::String("benchmark_baseline_not_run".into()),
            ),
            ("message", JsonValue::String("baseline missing".into())),
            ("reason", JsonValue::Null),
            ("severity", JsonValue::String("info".into())),
        ])]),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("warnings[].severity")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        &mut result,
        "warnings",
        JsonValue::Array(vec![object_from_pairs([
            (
                "code",
                JsonValue::String("benchmark_baseline_not_run".into()),
            ),
            ("message", JsonValue::String("baseline missing".into())),
            ("reason", JsonValue::Null),
        ])]),
    );
    validate_benchmark_result_contract(&result).unwrap();
}

#[test]
fn benchmark_result_contract_validates_engine_params() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result.as_object_mut().unwrap().remove("engine_params");
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("engine_params")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("engine_params")
        .and_then(JsonValue::as_object_mut)
        .and_then(|params| params.get_mut("text_index"))
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("analyzer".into(), JsonValue::String("standard".into()));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("engine_params.text_index.analyzer")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("engine_params")
        .and_then(JsonValue::as_object_mut)
        .and_then(|params| params.get_mut("text_index"))
        .and_then(JsonValue::as_object_mut)
        .and_then(|text| text.get_mut("bm25"))
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("k1".into(), JsonValue::Number(0.0));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("engine_params.text_index.bm25.k1")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("engine_params")
        .and_then(JsonValue::as_object_mut)
        .and_then(|params| params.get_mut("fusion"))
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("algorithm".into(), JsonValue::String("rrf".into()));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("engine_params.fusion.algorithm")
    );
}

#[test]
fn benchmark_result_contract_validates_profile_metrics_and_import_schema() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    result
        .as_object_mut()
        .unwrap()
        .get_mut("profile_params")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .remove("fusion_candidates");
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("profile_params.fusion_candidates")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("profile_params")
            .unwrap(),
        "disk_cache_mb",
        JsonValue::Number(64.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("profile_params.disk_cache_mb")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("profile_params")
            .unwrap(),
        "cache_budget_mb",
        JsonValue::Number(0.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("profile_params.cache_budget_mb")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("candidate_counts")
            .unwrap(),
        "vector",
        JsonValue::Number(0.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("candidate_counts.vector")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("candidate_counts")
            .unwrap(),
        "bm25",
        JsonValue::Number(999.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("candidate_counts.bm25")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("candidate_counts")
            .unwrap(),
        "rrf",
        JsonValue::Number(100.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("candidate_counts.rrf")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("metrics").unwrap(),
        "p95_ms",
        JsonValue::Number(-1.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("metrics.p95_ms")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result.as_object_mut().unwrap().get_mut("metrics").unwrap(),
        "avg_ms",
        JsonValue::Number(1.0),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("metrics.avg_ms")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("import_context")
            .unwrap(),
        "index_build_mode",
        JsonValue::String("latest_batch_only".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("import_context.index_build_mode")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("import_context")
            .unwrap(),
        "input_path",
        JsonValue::String("dataset".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("import_context.input_path")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("import_context")
            .unwrap(),
        "import_command",
        JsonValue::String(String::new()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("import_context.import_command")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("import_context")
            .unwrap(),
        "manifest_path",
        JsonValue::String(String::new()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("import_context.manifest_path")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        result
            .as_object_mut()
            .unwrap()
            .get_mut("import_context")
            .unwrap(),
        "on_conflict",
        JsonValue::String("merge".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("import_context.on_conflict")
    );
}

#[test]
fn benchmark_result_contract_requires_nonempty_per_query() {
    let mut run = benchmark_contract_run("text_bm25");
    insert_object_field(&mut run, "per_query", JsonValue::Array(Vec::new()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query")
    );

    let mut baseline_run = benchmark_contract_baseline_run("warm");
    insert_object_field(&mut baseline_run, "per_query", JsonValue::Array(Vec::new()));
    let result = benchmark_contract_completed_baseline_result(vec![
        benchmark_contract_baseline_run("cold"),
        baseline_run,
    ]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].per_query")
    );
}

#[test]
fn benchmark_result_contract_validates_per_query_score_schema() {
    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0].as_object_mut().unwrap().remove("score_kind");
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].score_kind")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .insert("score_scale".into(), JsonValue::String("global".into()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].score_scale")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .insert("score_comparable".into(), JsonValue::Bool(true));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].score_comparable")
    );
}

#[test]
fn benchmark_result_contract_validates_run_metrics_schema() {
    let mut run = benchmark_contract_run("text_bm25");
    run.as_object_mut()
        .unwrap()
        .get_mut("metrics")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .remove("p95_ms");
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].metrics.p95_ms")
    );

    let mut run = benchmark_contract_run("text_bm25");
    insert_object_field(&mut run, "profile", JsonValue::String("balanced".into()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].profile")
    );

    let mut run = benchmark_contract_run("text_bm25");
    run.as_object_mut()
        .unwrap()
        .get_mut("metrics")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("avg_ms".into(), JsonValue::Number(1.0));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].metrics.avg_ms")
    );
}

#[test]
fn benchmark_result_contract_validates_sidecar_run_schema() {
    let mut run = benchmark_contract_run("hybrid");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0].as_object_mut().unwrap().remove("score_kind");
    let result = benchmark_contract_result("sidecar", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].score_kind")
    );
}

#[test]
fn benchmark_result_contract_validates_per_query_score_alignment() {
    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    insert_object_field(
        &mut items[0],
        "scores",
        JsonValue::Array(vec![JsonValue::Number(1.0), JsonValue::Number(0.5)]),
    );
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].scores")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    insert_object_field(
        &mut items[0],
        "ranked_chunk_ids",
        JsonValue::Array(vec![
            JsonValue::String("doc#1".into()),
            JsonValue::String("doc#2".into()),
        ]),
    );
    insert_object_field(
        &mut items[0],
        "scores",
        JsonValue::Array(vec![JsonValue::Number(1.0), JsonValue::Number(0.5)]),
    );
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("top_k".into(), JsonValue::Number(1.0));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].ranked_chunk_ids")
    );
    assert_eq!(detail_value(&err, "top_k"), Some("1"));
    assert_eq!(detail_value(&err, "ranked_chunk_count"), Some("2"));
}

#[test]
fn benchmark_result_contract_validates_per_query_execution_stats() {
    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0].as_object_mut().unwrap().remove("execution_stats");
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].execution_stats")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("execution_stats")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("ann_probe_count".into(), JsonValue::Number(10.0));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].execution_stats.ann_probe_count")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("execution_stats")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("vector_scan_count".into(), JsonValue::String("one".into()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].execution_stats.vector_scan_count")
    );
}

#[test]
fn benchmark_result_contract_checks_effective_cofferdb_windows_without_constraining_baselines() {
    fn result_with_both_caches(mut run: JsonValue) -> JsonValue {
        insert_object_field(&mut run, "cache_state", JsonValue::String("cold".into()));
        let mut warm = run.clone();
        insert_object_field(&mut warm, "cache_state", JsonValue::String("warm".into()));
        benchmark_contract_result("inline", vec![run, warm], Vec::new())
    }
    for profile in [
        ResourceProfile::Constrained,
        ResourceProfile::Balanced,
        ResourceProfile::Expanded,
    ] {
        for top_k in [10, profile.params().bm25_candidates, 600] {
            let limit = profile.params().bm25_candidates.max(top_k) as f64;
            let mut valid = benchmark_contract_run("hybrid");
            let JsonValue::Array(rows) =
                valid.as_object_mut().unwrap().get_mut("per_query").unwrap()
            else {
                panic!("per_query");
            };
            let context = rows[0]
                .as_object_mut()
                .unwrap()
                .get_mut("query_context")
                .unwrap();
            insert_object_field(
                context,
                "profile",
                JsonValue::String(profile.as_json_str().into()),
            );
            insert_object_field(context, "top_k", JsonValue::Number(top_k as f64));
            insert_object_field(
                context,
                "candidate_limits",
                object_from_pairs([
                    ("bm25", JsonValue::Number(limit)),
                    ("vector", JsonValue::Number(limit)),
                    ("fusion", JsonValue::Number(limit)),
                ]),
            );
            validate_benchmark_result_contract(&result_with_both_caches(valid.clone())).unwrap();
            for branch in ["bm25", "vector", "fusion"] {
                for wrong in [0.0, limit - 1.0, limit + 1.0, limit + 0.5] {
                    let mut invalid = valid.clone();
                    let JsonValue::Array(rows) = invalid
                        .as_object_mut()
                        .unwrap()
                        .get_mut("per_query")
                        .unwrap()
                    else {
                        panic!("per_query");
                    };
                    let limits = rows[0]
                        .as_object_mut()
                        .unwrap()
                        .get_mut("query_context")
                        .unwrap()
                        .as_object_mut()
                        .unwrap()
                        .get_mut("candidate_limits")
                        .unwrap();
                    insert_object_field(limits, branch, JsonValue::Number(wrong));
                    let error =
                        validate_benchmark_result_contract(&result_with_both_caches(invalid))
                            .unwrap_err();
                    assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
                    assert_eq!(
                        detail_value(&error, "field"),
                        Some(
                            format!("runs[].per_query[].query_context.candidate_limits.{branch}")
                                .as_str()
                        )
                    );
                }
            }
        }
    }
    // An external engine may keep its own candidate policy.
    let mut baseline = benchmark_contract_baseline_run("cold");
    let JsonValue::Array(rows) = baseline
        .as_object_mut()
        .unwrap()
        .get_mut("per_query")
        .unwrap()
    else {
        panic!("per_query");
    };
    let context = rows[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .unwrap();
    insert_object_field(context, "top_k", JsonValue::Number(10.0));
    insert_object_field(
        context,
        "candidate_limits",
        object_from_pairs([
            ("bm25", JsonValue::Number(50.0)),
            ("vector", JsonValue::Number(50.0)),
            ("fusion", JsonValue::Number(50.0)),
        ]),
    );
    validate_benchmark_result_contract(&benchmark_contract_completed_baseline_result(vec![
        baseline,
        benchmark_contract_baseline_run("warm"),
    ]))
    .unwrap();
}

#[test]
fn benchmark_result_contract_validates_per_query_query_context() {
    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0].as_object_mut().unwrap().remove("query_context");
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingRequiredField);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("profile".into(), JsonValue::String("balanced-v2".into()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.profile")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("query_embedding".into(), JsonValue::Array(Vec::new()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.query_embedding")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .and_then(|context| context.get_mut("weights"))
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("debug".into(), JsonValue::Bool(true));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.weights.debug")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .and_then(|context| context.get_mut("requested_weights"))
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("text".into(), JsonValue::Number(1.0));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.requested_weights.text")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .and_then(|context| context.get_mut("candidate_limits"))
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("ann".into(), JsonValue::Number(100.0));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.candidate_limits.ann")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .and_then(|context| context.get_mut("transient_rebuild"))
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("reason".into(), JsonValue::String("none".into()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.transient_rebuild.reason")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("index_source".into(), JsonValue::String("ad_hoc".into()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.index_source")
    );

    let mut run = benchmark_contract_run("vector_exact");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("embedding_present".into(), JsonValue::Bool(false));
    let result = benchmark_contract_result("sidecar", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].query_context.embedding_present")
    );
    assert_eq!(detail_value(&err, "query_mode"), Some("vector_exact"));
}

#[test]
fn benchmark_result_contract_validates_per_query_types() {
    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    insert_object_field(&mut items[0], "hit", JsonValue::String("yes".into()));
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].hit")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    insert_object_field(
        &mut items[0],
        "scores",
        JsonValue::Array(vec![JsonValue::String("not-a-score".into())]),
    );
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].scores")
    );

    let mut run = benchmark_contract_run("text_bm25");
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract test run must contain per_query array");
    };
    items[0]
        .as_object_mut()
        .unwrap()
        .insert("raw_score_components".into(), JsonValue::object());
    let result = benchmark_contract_result("none", vec![run], Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].per_query[].raw_score_components")
    );
}

#[test]
fn benchmark_result_contract_validates_cache_states() {
    let result = benchmark_contract_result(
        "none",
        vec![benchmark_contract_run("text_bm25")],
        Vec::new(),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("runs[].cache_state")
    );

    let mut result = benchmark_contract_result(
        "none",
        vec![
            benchmark_contract_run_with_cache("text_bm25", "cold"),
            benchmark_contract_run("text_bm25"),
        ],
        vec![benchmark_contract_baseline_config()],
    );
    insert_object_field(
        &mut result,
        "baseline_results",
        JsonValue::Array(vec![benchmark_contract_baseline_run("warm")]),
    );
    insert_object_field(
        &mut result,
        "baseline_status",
        JsonValue::String("completed".into()),
    );
    insert_object_field(&mut result, "baseline_status_reason", JsonValue::Null);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].cache_state")
    );
}

#[test]
fn benchmark_result_contract_validates_baseline_result_links() {
    let result = benchmark_contract_completed_baseline_result(vec![
        benchmark_contract_baseline_run("cold"),
        benchmark_contract_baseline_run("warm"),
    ]);
    validate_benchmark_result_contract(&result).unwrap();

    let mut run = benchmark_contract_baseline_run("warm");
    run.as_object_mut().unwrap().remove("name");
    let result = benchmark_contract_completed_baseline_result(vec![run]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].name")
    );

    let mut run = benchmark_contract_baseline_run("warm");
    insert_object_field(
        &mut run,
        "name",
        JsonValue::String("unknown-baseline".into()),
    );
    let result = benchmark_contract_completed_baseline_result(vec![run]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].name")
    );

    let mut run = benchmark_contract_baseline_run("warm");
    insert_object_field(&mut run, "engine", JsonValue::String("duckdb".into()));
    let result = benchmark_contract_completed_baseline_result(vec![run]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].engine")
    );

    let mut result = benchmark_contract_completed_baseline_result(vec![
        benchmark_contract_baseline_run("cold"),
        benchmark_contract_baseline_run("warm"),
    ]);
    let configs = result
        .as_object_mut()
        .unwrap()
        .get_mut("baseline_configs")
        .unwrap();
    let JsonValue::Array(configs) = configs else {
        panic!("benchmark contract result must contain baseline_configs array");
    };
    configs[0]
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("query_count".into(), JsonValue::Number(2.0));
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].per_query")
    );
    assert_eq!(detail_value(&err, "expected_query_count"), Some("2"));
    assert_eq!(detail_value(&err, "actual_query_count"), Some("1"));

    let result = benchmark_contract_completed_baseline_result(Vec::new());
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].name")
    );
}

#[test]
fn benchmark_result_contract_requires_p0_sqlite_vec_baseline_coverage() {
    let result = benchmark_contract_completed_vector_baseline_result(vec![
        benchmark_contract_vector_baseline_run("text_bm25", "cold"),
        benchmark_contract_vector_baseline_run("text_bm25", "warm"),
        benchmark_contract_vector_baseline_run("vector_exact", "cold"),
        benchmark_contract_vector_baseline_run("vector_exact", "warm"),
        benchmark_contract_vector_baseline_run("hybrid", "cold"),
        benchmark_contract_vector_baseline_run("hybrid", "warm"),
    ]);
    validate_benchmark_result_contract(&result).unwrap();

    let result = benchmark_contract_completed_vector_baseline_result(vec![
        benchmark_contract_vector_baseline_run("text_bm25", "cold"),
        benchmark_contract_vector_baseline_run("text_bm25", "warm"),
        benchmark_contract_vector_baseline_run("hybrid", "cold"),
        benchmark_contract_vector_baseline_run("hybrid", "warm"),
    ]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results[].query_mode")
    );
    assert_eq!(
        detail_value(&err, "baseline_name"),
        Some("sqlite-fts5-sqlite-vec")
    );
    assert_eq!(
        detail_value(&err, "expected_query_mode"),
        Some("vector_exact")
    );

    let mut bad_config = benchmark_contract_vector_baseline_config();
    insert_object_field(&mut bad_config, "top_k", JsonValue::Number(99.0));
    let mut result = benchmark_contract_result("sidecar", Vec::new(), vec![bad_config]);
    insert_object_field(
        &mut result,
        "baseline_results",
        JsonValue::Array(vec![
            benchmark_contract_vector_baseline_run("text_bm25", "cold"),
            benchmark_contract_vector_baseline_run("text_bm25", "warm"),
            benchmark_contract_vector_baseline_run("vector_exact", "cold"),
            benchmark_contract_vector_baseline_run("vector_exact", "warm"),
            benchmark_contract_vector_baseline_run("hybrid", "cold"),
            benchmark_contract_vector_baseline_run("hybrid", "warm"),
        ]),
    );
    insert_object_field(
        &mut result,
        "baseline_status",
        JsonValue::String("completed".into()),
    );
    insert_object_field(&mut result, "baseline_status_reason", JsonValue::Null);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].top_k")
    );
    assert_eq!(detail_value(&err, "expected_top_k"), Some("10"));
    assert_eq!(detail_value(&err, "actual_top_k"), Some("99"));

    let mut bad_config = benchmark_contract_vector_baseline_config();
    bad_config
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("vector_extension".into(), JsonValue::Null);
    let mut result = benchmark_contract_result("sidecar", Vec::new(), vec![bad_config]);
    insert_object_field(
        &mut result,
        "baseline_results",
        JsonValue::Array(vec![
            benchmark_contract_vector_baseline_run("text_bm25", "cold"),
            benchmark_contract_vector_baseline_run("text_bm25", "warm"),
            benchmark_contract_vector_baseline_run("vector_exact", "cold"),
            benchmark_contract_vector_baseline_run("vector_exact", "warm"),
            benchmark_contract_vector_baseline_run("hybrid", "cold"),
            benchmark_contract_vector_baseline_run("hybrid", "warm"),
        ]),
    );
    insert_object_field(
        &mut result,
        "baseline_status",
        JsonValue::String("completed".into()),
    );
    insert_object_field(&mut result, "baseline_status_reason", JsonValue::Null);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].config.vector_extension")
    );
}

#[test]
fn benchmark_result_contract_validates_baseline_status_reason() {
    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(&mut result, "baseline_status_reason", JsonValue::Null);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_status_reason")
    );

    insert_object_field(
        &mut result,
        "baseline_status",
        JsonValue::String("completed".into()),
    );
    insert_object_field(
        &mut result,
        "baseline_status_reason",
        JsonValue::String("stale reason".into()),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_status_reason")
    );

    let mut result = benchmark_contract_result("none", Vec::new(), Vec::new());
    insert_object_field(
        &mut result,
        "baseline_results",
        JsonValue::Array(vec![benchmark_contract_baseline_run("warm")]),
    );
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_results")
    );
    assert_eq!(detail_value(&err, "baseline_status"), Some("not_run"));
}

#[test]
fn benchmark_result_contract_validates_baseline_config_schema() {
    let valid_config = object_from_pairs([
        ("name", JsonValue::String("sqlite-fts5-text-only".into())),
        ("version", JsonValue::Null),
        ("run_mode", JsonValue::String("local_cli".into())),
        ("mode", JsonValue::String("local_cli".into())),
        ("requires_query_embedding", JsonValue::Bool(false)),
        ("top_k", JsonValue::Number(10.0)),
        (
            "cache_states",
            JsonValue::Array(vec![
                JsonValue::String("cold".into()),
                JsonValue::String("warm".into()),
            ]),
        ),
        (
            "config_source",
            JsonValue::String("benchmarks/baselines/sqlite/README.md".into()),
        ),
        ("checked_at", JsonValue::Null),
        (
            "config",
            object_from_pairs([
                ("engine", JsonValue::String("sqlite".into())),
                ("sqlite_cli", JsonValue::String("sqlite3".into())),
                ("fts", JsonValue::String("fts5".into())),
                ("fts_tokenizer", JsonValue::String("default".into())),
                ("fts_score", JsonValue::String("bm25(chunks_fts)".into())),
                ("vector_extension", JsonValue::Null),
                ("vector_mode", JsonValue::Null),
                ("fusion", JsonValue::String("text_only_bm25".into())),
                (
                    "filter",
                    JsonValue::String("metadata_filter_after_candidate_fetch".into()),
                ),
                ("query_count", JsonValue::Number(1.0)),
                ("status", JsonValue::String("not_run".into())),
            ]),
        ),
    ]);
    let result = benchmark_contract_result("none", Vec::new(), vec![valid_config.clone()]);
    validate_benchmark_result_contract(&result).unwrap();

    // A valid resource preset is still unknown in the SQLite config schema.
    for field in ["profile", "resource_profile"] {
        for value in ["balanced", "application-specific-preset"] {
            let mut extra_nested_config = valid_config.clone();
            let config = extra_nested_config
                .as_object_mut()
                .unwrap()
                .get_mut("config")
                .unwrap();
            insert_object_field(config, field, JsonValue::String(value.into()));
            let result = benchmark_contract_result("none", Vec::new(), vec![extra_nested_config]);
            let before = result.to_compact_string();
            let err = validate_benchmark_result_contract(&result).unwrap_err();
            assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema, "{field}/{value}");
            let expected_field = format!("baseline_configs[].config.{field}");
            assert_eq!(detail_value(&err, "field"), Some(expected_field.as_str()));
            assert_eq!(result.to_compact_string(), before, "{field}/{value}");
        }
    }

    let mut extra_field_config = valid_config.clone();
    insert_object_field(
        &mut extra_field_config,
        "display_name",
        JsonValue::String("SQLite".into()),
    );
    let result = benchmark_contract_result("none", Vec::new(), vec![extra_field_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].display_name")
    );

    let mut extra_nested_config = valid_config.clone();
    extra_nested_config
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .and_then(JsonValue::as_object_mut)
        .unwrap()
        .insert("sqlite_version".into(), JsonValue::String("3.45.0".into()));
    let result = benchmark_contract_result("none", Vec::new(), vec![extra_nested_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].config.sqlite_version")
    );

    let mut zero_top_k_config = valid_config.clone();
    insert_object_field(&mut zero_top_k_config, "top_k", JsonValue::Number(0.0));
    let result = benchmark_contract_result("none", Vec::new(), vec![zero_top_k_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].top_k")
    );

    let mut zero_query_count_config = valid_config.clone();
    let config = zero_query_count_config
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .unwrap();
    insert_object_field(config, "query_count", JsonValue::Number(0.0));
    let result = benchmark_contract_result("none", Vec::new(), vec![zero_query_count_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].config.query_count")
    );

    let mut mismatched_status_config = valid_config.clone();
    let config = mismatched_status_config
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .unwrap();
    insert_object_field(config, "status", JsonValue::String("completed".into()));
    let result = benchmark_contract_result("none", Vec::new(), vec![mismatched_status_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].config.status")
    );

    let mut missing_cli_config = valid_config.clone();
    missing_cli_config
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove("sqlite_cli");
    let result = benchmark_contract_result("none", Vec::new(), vec![missing_cli_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].config.sqlite_cli")
    );

    let mut missing_fusion_config = valid_config.clone();
    missing_fusion_config
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove("fusion");
    let result = benchmark_contract_result("none", Vec::new(), vec![missing_fusion_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].config.fusion")
    );

    let mut invalid_date_config = valid_config.clone();
    insert_object_field(
        &mut invalid_date_config,
        "checked_at",
        JsonValue::String("2026-99-99".into()),
    );
    let result = benchmark_contract_result("none", Vec::new(), vec![invalid_date_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].checked_at")
    );

    let mut unverified_completed_config = valid_config.clone();
    insert_object_field(
        &mut unverified_completed_config,
        "version",
        JsonValue::String("3.45.0".into()),
    );
    let mut result =
        benchmark_contract_result("none", Vec::new(), vec![unverified_completed_config]);
    insert_object_field(
        &mut result,
        "baseline_status",
        JsonValue::String("completed".into()),
    );
    insert_object_field(&mut result, "baseline_status_reason", JsonValue::Null);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].checked_at")
    );

    let invalid_config = object_from_pairs([
        ("requires_query_embedding", JsonValue::Bool(false)),
        (
            "config",
            object_from_pairs([
                ("engine", JsonValue::String("sqlite".into())),
                ("sqlite_cli", JsonValue::String("sqlite3".into())),
                ("fts", JsonValue::String("fts5".into())),
                ("fts_tokenizer", JsonValue::String("default".into())),
                ("fts_score", JsonValue::String("bm25(chunks_fts)".into())),
                ("fusion", JsonValue::String("text_only_bm25".into())),
                (
                    "filter",
                    JsonValue::String("metadata_filter_after_candidate_fetch".into()),
                ),
                ("query_count", JsonValue::Number(1.0)),
            ]),
        ),
    ]);
    let result = benchmark_contract_result("none", Vec::new(), vec![invalid_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].name")
    );

    let invalid_config = object_from_pairs([
        ("name", JsonValue::String("sqlite-fts5-text-only".into())),
        ("version", JsonValue::Null),
        ("run_mode", JsonValue::String("local_cli".into())),
        ("mode", JsonValue::String("local_cli".into())),
        ("requires_query_embedding", JsonValue::Bool(false)),
        ("top_k", JsonValue::Number(10.0)),
        (
            "cache_states",
            JsonValue::Array(vec![
                JsonValue::String("cold".into()),
                JsonValue::String("warm".into()),
            ]),
        ),
        (
            "config_source",
            JsonValue::String("benchmarks/baselines/sqlite/README.md".into()),
        ),
        ("checked_at", JsonValue::Null),
        ("config", JsonValue::Null),
    ]);
    let result = benchmark_contract_result("none", Vec::new(), vec![invalid_config]);
    let err = validate_benchmark_result_contract(&result).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        err.detail()
            .iter()
            .find(|(key, _)| key == "field")
            .map(|(_, value)| value.as_str()),
        Some("baseline_configs[].config")
    );
}

fn benchmark_contract_result(
    query_embedding_source: &str,
    runs: Vec<JsonValue>,
    baseline_configs: Vec<JsonValue>,
) -> JsonValue {
    let (query_embedding_model, query_embedding_dim, query_embeddings_sha256) =
        match query_embedding_source {
            "sidecar" => (
                JsonValue::String("test-embedding-model".into()),
                JsonValue::Number(3.0),
                JsonValue::String("query-embeddings-sha".into()),
            ),
            "inline" => (
                JsonValue::String("test-embedding-model".into()),
                JsonValue::Number(3.0),
                JsonValue::Null,
            ),
            _ => (JsonValue::Null, JsonValue::Null, JsonValue::Null),
        };
    let (embedding_model, embeddings_sha256) = match query_embedding_source {
        "sidecar" | "inline" => (
            JsonValue::String("test-embedding-model".into()),
            JsonValue::String("embeddings-sha".into()),
        ),
        _ => (JsonValue::Null, JsonValue::Null),
    };
    object_from_pairs([
        ("schema_version", JsonValue::String("0.1".into())),
        ("cofferdb_version", JsonValue::String("0.1.0".into())),
        ("git_commit", JsonValue::String("test".into())),
        ("file_format_version", JsonValue::Number(1.0)),
        (
            "hardware",
            object_from_pairs([
                ("cpu", JsonValue::String("test CPU, 4 cores".into())),
                ("memory_gb", JsonValue::Number(16.0)),
                ("storage", JsonValue::String("local-storage".into())),
                ("os", JsonValue::String("test OS 1.0".into())),
            ]),
        ),
        ("result_file", JsonValue::Null),
        ("dataset_id", JsonValue::String("bench".into())),
        (
            "dataset",
            object_from_pairs([
                ("dataset_id", JsonValue::String("bench".into())),
                ("tokenizer", crate::TokenizerConfig::default().to_json()),
                ("version", JsonValue::String("0.1".into())),
                ("chunking", JsonValue::Null),
                ("embedding_model", embedding_model),
                ("embedding_dim", JsonValue::Number(3.0)),
                (
                    "dataset_manifest_sha256",
                    JsonValue::String("manifest-sha".into()),
                ),
                ("documents_sha256", JsonValue::String("docs-sha".into())),
                ("chunks_sha256", JsonValue::String("chunks-sha".into())),
                ("embeddings_sha256", embeddings_sha256),
                ("queries_sha256", JsonValue::String("queries-sha".into())),
                ("qrels_sha256", JsonValue::String("qrels-sha".into())),
                (
                    "query_embedding_source",
                    JsonValue::String(query_embedding_source.into()),
                ),
                ("query_embedding_model", query_embedding_model),
                ("query_embedding_dim", query_embedding_dim),
                ("query_embeddings_sha256", query_embeddings_sha256),
            ]),
        ),
        ("profile", JsonValue::String("balanced".into())),
        (
            "profile_params",
            object_from_pairs([
                ("bm25_candidates", JsonValue::Number(200.0)),
                ("vector_candidates", JsonValue::Number(200.0)),
                ("fusion_candidates", JsonValue::Number(200.0)),
                ("cache_budget_mb", JsonValue::Number(64.0)),
            ]),
        ),
        ("engine_params", benchmark_engine_params_json()),
        (
            "candidate_counts",
            object_from_pairs([
                ("bm25", JsonValue::Number(200.0)),
                ("vector", JsonValue::Number(200.0)),
                ("fusion", JsonValue::Number(200.0)),
            ]),
        ),
        ("durability", JsonValue::String("commit".into())),
        (
            "index_build_mode",
            JsonValue::String("inline_full_checkpoint".into()),
        ),
        (
            "import_context",
            object_from_pairs([
                ("tokenizer", crate::TokenizerConfig::default().to_json()),
                ("batch_size", JsonValue::Number(1000.0)),
                (
                    "index_build_mode",
                    JsonValue::String("inline_full_checkpoint".into()),
                ),
                ("import_time_ms", JsonValue::Number(1.0)),
                (
                    "import_command",
                    JsonValue::String(
                        "coffer import <benchmark-db> --manifest <generated-import-manifest> --on-conflict replace --durability commit"
                            .into(),
                    ),
                ),
                (
                    "manifest_path",
                    JsonValue::String("<generated-import-manifest>".into()),
                ),
                ("database_path", JsonValue::String("<benchmark-db>".into())),
                ("on_conflict", JsonValue::String("replace".into())),
                ("summary", JsonValue::object()),
                ("durability", JsonValue::String("commit".into())),
            ]),
        ),
        (
            "metrics",
            object_from_pairs([
                ("recall_at_5", JsonValue::Number(0.0)),
                ("recall_at_10", JsonValue::Number(0.0)),
                ("mrr", JsonValue::Number(0.0)),
                ("p95_ms", JsonValue::Number(1.0)),
                ("memory_peak_mb", JsonValue::Null),
                ("vector_scan_count", JsonValue::Number(0.0)),
                ("distance_ops", JsonValue::Number(0.0)),
                ("fusion_candidate_count", JsonValue::Number(0.0)),
                ("import_time_ms", JsonValue::Number(1.0)),
                ("index_size_mb", JsonValue::Number(0.0)),
            ]),
        ),
        ("runs", JsonValue::Array(runs)),
        ("baseline_configs", JsonValue::Array(baseline_configs)),
        ("baseline_results", JsonValue::Array(Vec::new())),
        ("baseline_status", JsonValue::String("not_run".into())),
        (
            "baseline_status_reason",
            JsonValue::String("test baseline unavailable".into()),
        ),
    ])
}

fn benchmark_contract_run(query_mode: &str) -> JsonValue {
    benchmark_contract_run_with_cache(query_mode, "warm")
}

fn benchmark_contract_baseline_config() -> JsonValue {
    object_from_pairs([
        ("name", JsonValue::String("sqlite-fts5-text-only".into())),
        ("version", JsonValue::String("3.45.0".into())),
        ("run_mode", JsonValue::String("local_cli".into())),
        ("mode", JsonValue::String("local_cli".into())),
        ("requires_query_embedding", JsonValue::Bool(false)),
        ("top_k", JsonValue::Number(10.0)),
        (
            "cache_states",
            JsonValue::Array(vec![
                JsonValue::String("cold".into()),
                JsonValue::String("warm".into()),
            ]),
        ),
        (
            "config_source",
            JsonValue::String("benchmarks/baselines/sqlite/README.md".into()),
        ),
        ("checked_at", JsonValue::String("2026-05-31".into())),
        (
            "config",
            object_from_pairs([
                ("engine", JsonValue::String("sqlite".into())),
                ("sqlite_cli", JsonValue::String("sqlite3".into())),
                ("fts", JsonValue::String("fts5".into())),
                ("fts_tokenizer", JsonValue::String("default".into())),
                ("fts_score", JsonValue::String("bm25(chunks_fts)".into())),
                ("vector_extension", JsonValue::Null),
                ("vector_mode", JsonValue::Null),
                ("fusion", JsonValue::String("text_only_bm25".into())),
                (
                    "filter",
                    JsonValue::String("metadata_filter_after_candidate_fetch".into()),
                ),
                ("query_count", JsonValue::Number(1.0)),
                ("status", JsonValue::String("completed".into())),
            ]),
        ),
    ])
}

fn benchmark_contract_vector_baseline_config() -> JsonValue {
    let mut config = benchmark_contract_baseline_config();
    insert_object_field(
        &mut config,
        "name",
        JsonValue::String("sqlite-fts5-sqlite-vec".into()),
    );
    insert_object_field(
        &mut config,
        "requires_query_embedding",
        JsonValue::Bool(true),
    );
    let config_obj = config
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .and_then(JsonValue::as_object_mut)
        .unwrap();
    config_obj.insert(
        "vector_extension".into(),
        JsonValue::String("sqlite-vec".into()),
    );
    config_obj.insert("vector_mode".into(), JsonValue::String("exact_scan".into()));
    config_obj.insert("fusion".into(), JsonValue::String("weighted_sum".into()));
    config
}

fn benchmark_contract_baseline_run(cache_state: &str) -> JsonValue {
    benchmark_contract_named_baseline_run("sqlite-fts5-text-only", "text_bm25", cache_state)
}

fn benchmark_contract_vector_baseline_run(query_mode: &str, cache_state: &str) -> JsonValue {
    benchmark_contract_named_baseline_run("sqlite-fts5-sqlite-vec", query_mode, cache_state)
}

fn benchmark_contract_named_baseline_run(
    baseline_name: &str,
    query_mode: &str,
    cache_state: &str,
) -> JsonValue {
    let mut run = benchmark_contract_run_with_cache(query_mode, cache_state);
    insert_object_field(&mut run, "engine", JsonValue::String("sqlite".into()));
    insert_object_field(&mut run, "name", JsonValue::String(baseline_name.into()));
    let per_query = run.as_object_mut().unwrap().get_mut("per_query").unwrap();
    let JsonValue::Array(items) = per_query else {
        panic!("benchmark contract baseline run must contain per_query array");
    };
    insert_object_field(
        &mut items[0],
        "score_kind",
        JsonValue::String(
            match query_mode {
                "vector_exact" => "sqlite_vec_cosine",
                "hybrid" => "sqlite_fts5_vec_weighted_sum",
                _ => "sqlite_fts5_bm25",
            }
            .into(),
        ),
    );
    let context = items[0]
        .as_object_mut()
        .unwrap()
        .get_mut("query_context")
        .and_then(JsonValue::as_object_mut)
        .unwrap();
    context.insert("tokenizer".into(), JsonValue::Null);
    context.insert(
        "index_source".into(),
        JsonValue::String(
            sqlite_index_source(match query_mode {
                "vector_exact" => BenchmarkQueryMode::VectorExact,
                "hybrid" => BenchmarkQueryMode::Hybrid,
                _ => BenchmarkQueryMode::TextBm25,
            })
            .into(),
        ),
    );
    run
}

fn benchmark_contract_completed_baseline_result(baseline_results: Vec<JsonValue>) -> JsonValue {
    let mut result = benchmark_contract_result(
        "none",
        Vec::new(),
        vec![benchmark_contract_baseline_config()],
    );
    insert_object_field(
        &mut result,
        "baseline_results",
        JsonValue::Array(baseline_results),
    );
    insert_object_field(
        &mut result,
        "baseline_status",
        JsonValue::String("completed".into()),
    );
    insert_object_field(&mut result, "baseline_status_reason", JsonValue::Null);
    result
}

fn benchmark_contract_completed_vector_baseline_result(
    baseline_results: Vec<JsonValue>,
) -> JsonValue {
    let mut result = benchmark_contract_result(
        "sidecar",
        Vec::new(),
        vec![benchmark_contract_vector_baseline_config()],
    );
    insert_object_field(
        &mut result,
        "baseline_results",
        JsonValue::Array(baseline_results),
    );
    insert_object_field(
        &mut result,
        "baseline_status",
        JsonValue::String("completed".into()),
    );
    insert_object_field(&mut result, "baseline_status_reason", JsonValue::Null);
    result
}

fn benchmark_contract_run_with_cache(query_mode: &str, cache_state: &str) -> JsonValue {
    let score_kind = match query_mode {
        "vector_exact" => "cofferdb_vector_final_score",
        "hybrid" => "cofferdb_hybrid_final_score",
        _ => "cofferdb_bm25_final_score",
    };
    object_from_pairs([
        ("engine", JsonValue::String("cofferdb".into())),
        ("query_mode", JsonValue::String(query_mode.into())),
        ("cache_state", JsonValue::String(cache_state.into())),
        (
            "metrics",
            object_from_pairs([
                ("recall_at_5", JsonValue::Number(0.0)),
                ("recall_at_10", JsonValue::Number(0.0)),
                ("mrr", JsonValue::Number(0.0)),
                ("p95_ms", JsonValue::Number(1.0)),
                ("memory_peak_mb", JsonValue::Null),
                ("vector_scan_count", JsonValue::Number(0.0)),
                ("distance_ops", JsonValue::Number(0.0)),
                ("fusion_candidate_count", JsonValue::Number(0.0)),
            ]),
        ),
        (
            "per_query",
            JsonValue::Array(vec![object_from_pairs([
                ("query_id", JsonValue::String("q1".into())),
                ("latency_ms", JsonValue::Number(1.0)),
                (
                    "ranked_chunk_ids",
                    JsonValue::Array(vec![JsonValue::String("doc#1".into())]),
                ),
                ("score_kind", JsonValue::String(score_kind.into())),
                ("score_scale", JsonValue::String("engine_local".into())),
                ("score_comparable", JsonValue::Bool(false)),
                ("scores", JsonValue::Array(vec![JsonValue::Number(1.0)])),
                ("hit", JsonValue::Bool(true)),
                (
                    "execution_stats",
                    BenchmarkExecutionStats::default().to_json(),
                ),
                (
                    "query_context",
                    benchmark_contract_query_context_for_mode(query_mode),
                ),
            ])]),
        ),
    ])
}

fn benchmark_contract_query_context_for_mode(query_mode: &str) -> JsonValue {
    let (bm25_weight, vector_weight) = match query_mode {
        "vector_exact" => (0.0, 1.0),
        "hybrid" => (0.5, 0.5),
        _ => (1.0, 0.0),
    };
    let weights = object_from_pairs([
        ("bm25", JsonValue::Number(bm25_weight)),
        ("vector", JsonValue::Number(vector_weight)),
        ("metadata", JsonValue::Number(0.0)),
    ]);
    object_from_pairs([
        (
            "text_present",
            JsonValue::Bool(matches!(query_mode, "text_bm25" | "hybrid")),
        ),
        (
            "tokenizer",
            if matches!(query_mode, "text_bm25" | "hybrid") {
                crate::TokenizerConfig::default().to_json()
            } else {
                JsonValue::Null
            },
        ),
        (
            "embedding_present",
            JsonValue::Bool(matches!(query_mode, "vector_exact" | "hybrid")),
        ),
        ("filter_present", JsonValue::Bool(false)),
        (
            "weights",
            object_from_pairs([
                ("present", JsonValue::Bool(false)),
                ("source", JsonValue::String("default_no_boost".into())),
                ("requested", weights.clone()),
                ("effective", weights.clone()),
                ("renormalization_reason", JsonValue::Null),
            ]),
        ),
        ("requested_weights", weights.clone()),
        ("effective_weights", weights),
        (
            "weights_source",
            JsonValue::String("default_no_boost".into()),
        ),
        ("renormalization_reason", JsonValue::Null),
        ("metadata_boost_present", JsonValue::Bool(false)),
        ("metadata_boost_enabled", JsonValue::Bool(false)),
        ("boost_config_hash", JsonValue::Null),
        ("boost_rule_count", JsonValue::Number(0.0)),
        ("top_k", JsonValue::Number(10.0)),
        ("profile", JsonValue::String("balanced".into())),
        (
            "candidate_limits",
            object_from_pairs([
                ("bm25", JsonValue::Number(200.0)),
                ("vector", JsonValue::Number(200.0)),
                ("fusion", JsonValue::Number(200.0)),
            ]),
        ),
        (
            "execution_stats",
            BenchmarkExecutionStats::default().to_json(),
        ),
        (
            "index_source",
            JsonValue::String("persistent_checkpoint".into()),
        ),
        ("index_rebuild_reason", JsonValue::Null),
        (
            "transient_rebuild",
            object_from_pairs([
                ("started", JsonValue::Bool(false)),
                ("completed", JsonValue::Bool(false)),
                ("cancelled", JsonValue::Bool(false)),
                ("budget_mb", JsonValue::Number(256.0)),
                ("chunks_scanned", JsonValue::Null),
                ("bytes_read", JsonValue::Null),
                ("budget_exceeded", JsonValue::Bool(false)),
            ]),
        ),
    ])
}

#[test]
fn benchmark_metrics_json_records_memory_peak_when_available() {
    let metrics = BenchmarkRunMetrics {
        memory_peak_mb: Some(12.5),
        ..BenchmarkRunMetrics::default()
    };
    let json = benchmark_metrics_json(&metrics);
    assert_eq!(
        json.get("memory_peak_mb").and_then(JsonValue::as_f64),
        Some(12.5)
    );
    if let Some(memory_mb) = process_memory_peak_mb() {
        assert!(memory_mb > 0.0);
    }
    assert_eq!(
        benchmark_metrics_json(&BenchmarkRunMetrics::default()).get("memory_peak_mb"),
        Some(&JsonValue::Null)
    );
}
