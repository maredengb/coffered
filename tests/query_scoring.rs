use std::fs;
use std::path::PathBuf;

use cofferdb::{
    parse_json, CheckOptions, ChunkInput, CollectionConfig, CollectionHandle, Database,
    DatabaseConfig, DatabaseOpenOptions, DistanceMetric, DocumentInput, EmbeddingInput, ErrorCode,
    JsonValue, MetadataBoostConfig, QueryEmbeddingInput, ResourceProfile, SearchWeights,
    VectorIndexConfig,
};

struct Fixture(PathBuf);

impl Fixture {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "cofferdb-scoring-{name}-{}.coffer",
            std::process::id()
        ));
        assert!(!path.exists(), "stale fixture: {}", path.display());
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn collection_config(name: String, distance_metric: DistanceMetric) -> CollectionConfig {
    CollectionConfig {
        name,
        embedding_dim: 2,
        distance_metric,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
        tokenizer: Default::default(),
    }
}

fn rank(value: &JsonValue, field: &str) -> Option<usize> {
    value.get(field).unwrap().as_f64().map(|n| n as usize)
}

fn insert_tied_chunks(collection: &CollectionHandle) {
    collection
        .insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..75)
                .map(|i| ChunkInput {
                    chunk_external_id: format!("chunk-{i}"),
                    text: "needle".into(),
                    ordinal: i,
                    metadata: parse_json(&format!(r#"{{"index":{i}}}"#)).unwrap(),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
}

#[test]
fn metadata_boost_uses_configured_ceiling_in_ranking_scores_and_explain() {
    let fixture = Fixture::new("metadata-boost-ceiling");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
        .unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..2)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    text: "needle".into(),
                    ordinal,
                    metadata: parse_json(&format!(r#"{{"index":{ordinal}}}"#)).unwrap(),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
    collection
        .put_embeddings(
            (0..2)
                .map(|ordinal| EmbeddingInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    model_name: "test".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .collect(),
        )
        .unwrap();
    let ids =
        ["chunk-0", "chunk-1"].map(|id| collection.get_chunk_by_external_id(id).unwrap().chunk_id);
    let critical_end = db
        .check(CheckOptions::default())
        .unwrap()
        .critical_end_offset;
    drop(collection);
    drop(db);
    for source in ["persistent_checkpoint", "transient_rebuild"] {
        if source == "transient_rebuild" {
            fs::OpenOptions::new()
                .write(true)
                .open(&fixture.0)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
        }
        let before = fs::read(&fixture.0).unwrap();
        let db = Database::open_read_only(&fixture.0).unwrap();
        let collection = db.collection("notes").unwrap();
        for ceiling in [2.0, 1.5, 0.2, 0.0] {
            let boost = MetadataBoostConfig::parse(parse_json(&format!(r#"{{"schema_version":"0.1","combine":"sum","max_boost":{ceiling},"rules":[{{"when":{{"chunk.index":0}},"boost":1.25}},{{"when":{{"chunk.index":1}},"boost":1.75}}]}}"#)).unwrap()).unwrap();
            let mut expected = [(0, 1.25_f64.min(ceiling)), (1, 1.75_f64.min(ceiling))];
            expected.sort_by(|(a, boost_a), (b, boost_b)| {
                boost_b
                    .total_cmp(boost_a)
                    .then_with(|| ids[*a].cmp(&ids[*b]))
            });
            for profile in [
                ResourceProfile::Expanded,
                ResourceProfile::Balanced,
                ResourceProfile::Constrained,
            ] {
                for hybrid in [false, true] {
                    for top_k in [1, 2] {
                        let mut query = collection
                            .search()
                            .text("needle")
                            .metadata_boost(boost.clone())
                            .resource_profile(profile)
                            .top_k(top_k)
                            .weights(
                                SearchWeights::new(
                                    if hybrid { 0.25 } else { 0.5 },
                                    if hybrid { 0.25 } else { 0.0 },
                                    0.5,
                                )
                                .unwrap(),
                            );
                        if hybrid {
                            query = query
                                .query_embedding(QueryEmbeddingInput::from_vector(vec![1.0, 0.0]));
                        }
                        let results = query.execute().unwrap();
                        assert_eq!(results.query_context.index_source, source);
                        assert_eq!(results.results.len(), top_k);
                        for (rank, (hit, (ordinal, boost))) in
                            results.results.iter().zip(expected).enumerate()
                        {
                            let score = 0.5 + 0.5 * boost;
                            assert_eq!(hit.chunk_external_id, format!("chunk-{ordinal}"));
                            assert_eq!(hit.rank, rank + 1);
                            assert_eq!(hit.metadata_boost, boost);
                            assert_eq!(hit.normalized_scores.metadata_boost, boost);
                            assert_eq!(hit.final_score, score);
                            assert_eq!(
                                hit.explain
                                    .get("scores")
                                    .unwrap()
                                    .get("metadata_boost")
                                    .unwrap()
                                    .as_f64(),
                                Some(boost)
                            );
                            assert_eq!(
                                hit.explain
                                    .get("scores")
                                    .unwrap()
                                    .get("final_score")
                                    .unwrap()
                                    .as_f64(),
                                Some(score)
                            );
                            let wire = hit.to_json();
                            assert_eq!(wire.get("metadata_boost").unwrap().as_f64(), Some(boost));
                            assert_eq!(
                                wire.get("normalized_scores")
                                    .unwrap()
                                    .get("metadata_boost")
                                    .unwrap()
                                    .as_f64(),
                                Some(boost)
                            );
                            assert_eq!(wire.get("final_score").unwrap().as_f64(), Some(score));
                        }
                    }
                }
            }
        }
        assert_eq!(fs::read(&fixture.0).unwrap(), before);
    }
}

#[test]
fn transient_text_search_scales_with_query_candidates_instead_of_full_postings() {
    let fixture = Fixture::new("bounded-transient-text");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
        .unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..2_200)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    text: if ordinal % 2 == 0 {
                        "common gamma"
                    } else {
                        "common delta"
                    }
                    .into(),
                    ordinal,
                    metadata: JsonValue::object(),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
    let queries = [
        "gamma",
        "common",
        "delta",
        "gamma delta",
        "absent",
        "the and",
    ];
    let expected = queries
        .iter()
        .map(|query| {
            collection
                .search()
                .text(*query)
                .top_k(10)
                .execute()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let critical_end = db
        .check(CheckOptions::default())
        .unwrap()
        .critical_end_offset;
    drop(collection);
    drop(db);
    fs::OpenOptions::new()
        .write(true)
        .open(&fixture.0)
        .unwrap()
        .set_len(critical_end)
        .unwrap();
    let original = fs::read(&fixture.0).unwrap();
    let reader = Database::open_with_options(
        &fixture.0,
        DatabaseOpenOptions {
            read_only: true,
            cache_budget_mb: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    for (query, expected) in queries.into_iter().zip(expected) {
        let actual = reader
            .collection("notes")
            .unwrap()
            .search()
            .text(query)
            .top_k(10)
            .execute()
            .unwrap();
        assert_eq!(actual.query_context.index_source, "transient_rebuild");
        assert_eq!(actual.query_context.transient_rebuild.budget_mb, 1);
        assert_eq!(actual.results.len(), expected.results.len());
        for (actual, expected) in actual.results.iter().zip(expected.results) {
            assert_eq!(actual.chunk_id, expected.chunk_id);
            assert_eq!(actual.bm25_score, expected.bm25_score);
            assert_eq!(actual.final_score, expected.final_score);
            for field in [
                "matched_terms",
                "matched_term_details",
                "bm25",
                "candidate_sources",
                "normalization",
            ] {
                assert_eq!(
                    actual.explain.get(field),
                    expected.explain.get(field),
                    "query={query}, field={field}"
                );
            }
        }
    }
    let retry_collection = reader.collection("notes").unwrap();
    let small_query = || {
        retry_collection
            .search()
            .text("gamma delta")
            .resource_profile(ResourceProfile::Constrained)
            .top_k(10)
            .execute()
    };
    let baseline = {
        let result = small_query().unwrap();
        assert_eq!(result.query_context.transient_rebuild.budget_mb, 1);
        result
            .results
            .iter()
            .map(|hit| (hit.chunk_id.clone(), hit.bm25_score, hit.final_score))
            .collect::<Vec<_>>()
    };
    for (query, top_k) in [
        ("gamma delta".to_owned(), 600),
        ("gamma delta".to_owned(), 2_000),
        ("x".repeat(8_192), 1),
    ] {
        let error = retry_collection
            .search()
            .text(query)
            .resource_profile(ResourceProfile::Constrained)
            .top_k(top_k)
            .execute()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IndexCorruption);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "index_unavailable_reason"
                && value == "transient_rebuild_budget_exceeded"));
        let detail = |key: &str| {
            error
                .detail()
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        assert_eq!(detail("budget_mb"), Some("1"));
        assert_eq!(detail("resource_profile"), Some("constrained"));
        assert!(detail("bytes_required").unwrap().parse::<u64>().unwrap() > 1024 * 1024);
        let retry = small_query().unwrap();
        assert_eq!(retry.query_context.transient_rebuild.budget_mb, 1);
        assert_eq!(retry.query_context.candidate_limits.bm25, 50);
        assert_eq!(
            retry
                .results
                .iter()
                .map(|hit| (hit.chunk_id.clone(), hit.bm25_score, hit.final_score))
                .collect::<Vec<_>>(),
            baseline
        );
        assert_eq!(fs::read(&fixture.0).unwrap(), original);
    }
    drop(retry_collection);
    let error = reader
        .why_not_debug(
            "notes",
            "chunk-1000",
            Some("gamma".into()),
            None,
            JsonValue::object(),
            None,
            false,
            10,
            Some(ResourceProfile::Expanded),
        )
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::IndexCorruption);
    assert!(error
        .detail()
        .iter()
        .any(|(key, value)| key == "index_unavailable_reason"
            && value == "transient_rebuild_budget_exceeded"));
    let diagnostic_reader = Database::open_with_options(
        &fixture.0,
        DatabaseOpenOptions {
            read_only: true,
            cache_budget_mb: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    let diagnosis = diagnostic_reader
        .why_not_debug(
            "notes",
            "chunk-1000",
            Some("gamma".into()),
            None,
            JsonValue::object(),
            None,
            false,
            10,
            None,
        )
        .unwrap();
    assert!(diagnosis.debug_full_scan);
    assert_eq!(diagnosis.query_context.transient_rebuild.budget_mb, 2);
    assert_eq!(
        rank(
            diagnosis.result.get("candidate_ranks").unwrap(),
            "bm25_rank"
        ),
        Some(501)
    );
    assert_eq!(fs::read(&fixture.0).unwrap(), original);
}

#[test]
fn repeated_tokens_keep_counts_and_bm25_scores_across_persistent_and_transient_indexes() {
    let fixture = Fixture::new("streamed-tokens");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("tokens".into(), DistanceMetric::Cosine))
        .unwrap();
    let repeats = 5_000;
    collection
        .insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: [
                "AlphaBeta alpha_beta THE _alpha_alpha_ 42 ".repeat(repeats),
                "alpha".into(),
                "THE and with".into(),
            ]
            .into_iter()
            .enumerate()
            .map(|(ordinal, text)| ChunkInput {
                chunk_external_id: format!("chunk-{ordinal}"),
                text,
                ordinal,
                metadata: JsonValue::object(),
                source_span: None,
            })
            .collect(),
        })
        .unwrap();
    let critical_end = db
        .check(CheckOptions::default())
        .unwrap()
        .critical_end_offset;
    drop(collection);
    drop(db);
    let query = "ALPHA alpha ".repeat(1_000);
    let avg_len = (repeats * 9 + 2) as f64 / 3.0;
    let idf = 1.6_f64.ln();
    for source in ["persistent_checkpoint", "transient_rebuild"] {
        if source == "transient_rebuild" {
            fs::OpenOptions::new()
                .write(true)
                .open(&fixture.0)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
        }
        let db = Database::open_read_only(&fixture.0).unwrap();
        let collection = db.collection("tokens").unwrap();
        for (ordinal, count) in [repeats * 9, 1, 0].into_iter().enumerate() {
            let chunk = collection
                .get_chunk_by_external_id(&format!("chunk-{ordinal}"))
                .unwrap();
            assert_eq!(chunk.token_count, count);
        }
        let result = collection
            .search()
            .text(&query)
            .top_k(10)
            .execute()
            .unwrap();
        assert_eq!(result.query_context.index_source, source);
        assert_eq!(result.results.len(), 2);
        for hit in result.results {
            let (frequency, length) = match hit.chunk_external_id.as_str() {
                "chunk-0" => ((repeats * 3) as f64, (repeats * 9) as f64),
                "chunk-1" => (1.0, 1.0),
                id => panic!("unexpected result {id}"),
            };
            let expected =
                idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / avg_len));
            let raw = hit
                .explain
                .get("scores")
                .unwrap()
                .get("bm25_raw")
                .unwrap()
                .as_f64()
                .unwrap();
            assert!(
                (raw - expected).abs() < 1e-12,
                "{source}: {raw} != {expected}"
            );
        }
        assert!(collection
            .search()
            .text("the and")
            .execute()
            .unwrap()
            .results
            .is_empty());
    }
}

#[test]
fn candidate_cutoffs_match_an_independent_oracle_across_modes_profiles_and_sources() {
    use std::collections::BTreeMap;

    struct ExpectedScore<'a> {
        id: &'a str,
        text_raw: Option<f64>,
        vector_raw: Option<f64>,
        text_norm: f64,
        vector_norm: Option<f64>,
        final_score: f64,
    }

    const COUNT: usize = 640;
    let frequency = |ordinal: usize| match ordinal % 11 {
        0 => 0,
        _ if ordinal % 3 == 0 => 2,
        _ => 1,
    };
    let vector = |ordinal: usize| {
        (ordinal % 19 != 0).then_some([(ordinal % 31 + 1) as f32, (ordinal % 17 + 1) as f32])
    };
    let fixture = Fixture::new("hybrid-cutoff-oracle");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
        .unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..COUNT)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    text: match frequency(ordinal) {
                        0 => "hay other",
                        1 => "needle hay",
                        _ => "needle needle hay",
                    }
                    .into(),
                    ordinal,
                    metadata: parse_json(&format!(r#"{{"eligible":{}}}"#, ordinal % 7 != 0))
                        .unwrap(),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
    collection
        .put_embeddings(
            (0..COUNT)
                .filter_map(|ordinal| {
                    vector(ordinal).map(|value| EmbeddingInput {
                        chunk_external_id: format!("chunk-{ordinal}"),
                        model_name: "fixture".into(),
                        dim: 2,
                        vector: value.to_vec(),
                    })
                })
                .collect(),
        )
        .unwrap();
    let chunks = collection
        .list_chunks_by_document_external_id("doc")
        .unwrap();
    let df = (0..COUNT)
        .filter(|&ordinal| frequency(ordinal) != 0)
        .count();
    let average_length = (0..COUNT)
        .map(|ordinal| if frequency(ordinal) == 2 { 3 } else { 2 })
        .sum::<usize>() as f64
        / COUNT as f64;
    let idf = ((COUNT as f64 + 1.0) / (df as f64 + 0.5)).ln();
    let text_score = |ordinal| {
        let tf = frequency(ordinal) as f64;
        let length = if tf == 2.0 { 3.0 } else { 2.0 };
        (tf > 0.0).then_some(idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average_length)))
    };
    let vector_score = |ordinal| {
        vector(ordinal).map(|[x, y]| {
            let (x, y) = (f64::from(x), f64::from(y));
            x / (x * x + y * y).sqrt()
        })
    };
    let critical_end = db
        .check(CheckOptions::default())
        .unwrap()
        .critical_end_offset;
    drop(collection);
    db.close().unwrap();

    for source in ["persistent_checkpoint", "transient_rebuild"] {
        let mut saw_missing_embedding = false;
        let mut saw_vector_only = false;
        if source == "transient_rebuild" {
            fs::OpenOptions::new()
                .write(true)
                .open(&fixture.0)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
        }
        let before = fs::read(&fixture.0).unwrap();
        let db = Database::open_read_only(&fixture.0).unwrap();
        let collection = db.collection("notes").unwrap();
        for (profile, default_limit) in [
            (ResourceProfile::Constrained, 50),
            (ResourceProfile::Balanced, 200),
            (ResourceProfile::Expanded, 500),
        ] {
            for top_k in [default_limit - 1, default_limit, default_limit + 1, 600] {
                let limit = default_limit.max(top_k);
                for (has_text, has_vector) in [(true, true), (true, false), (false, true)] {
                    for filtered in [false, true] {
                        let eligible =
                            |chunk: &cofferdb::Chunk| !filtered || chunk.ordinal % 7 != 0;
                        // Full sorting and explicit arithmetic do not reuse the production heaps or fuse.
                        let mut text = chunks
                            .iter()
                            .filter_map(|chunk| {
                                text_score(chunk.ordinal).map(|score| (chunk, score))
                            })
                            .collect::<Vec<_>>();
                        text.sort_by(|(a, a_score), (b, b_score)| {
                            b_score
                                .total_cmp(a_score)
                                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
                        });
                        if !has_text {
                            text.clear();
                        }
                        text.truncate(limit);
                        text.retain(|(chunk, _)| eligible(chunk));
                        let mut vectors = chunks
                            .iter()
                            .filter(|chunk| eligible(chunk))
                            .filter_map(|chunk| {
                                vector_score(chunk.ordinal).map(|score| (chunk, score))
                            })
                            .collect::<Vec<_>>();
                        let scanned = vectors.len();
                        assert!(scanned > 500 && df > 500);
                        vectors.sort_by(|(a, a_score), (b, b_score)| {
                            ((b_score + 1.0) / 2.0)
                                .total_cmp(&((a_score + 1.0) / 2.0))
                                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
                        });
                        if !has_vector {
                            vectors.clear();
                        }
                        vectors.truncate(limit);
                        let text_count = text.len();
                        let vector_count = vectors.len();
                        let supplemental_vectors = text
                            .iter()
                            .filter(|(chunk, _)| {
                                has_vector
                                    && vector(chunk.ordinal).is_some()
                                    && !vectors
                                        .iter()
                                        .any(|(row, _)| row.chunk_id == chunk.chunk_id)
                            })
                            .count();
                        let total_scans = if has_vector {
                            (0..COUNT)
                                .filter(|&ordinal| vector(ordinal).is_some())
                                .count()
                                + supplemental_vectors
                        } else {
                            0
                        };
                        let total_distance_vectors = if has_vector {
                            scanned + supplemental_vectors
                        } else {
                            0
                        };
                        let mut admitted = BTreeMap::new();
                        for (chunk, _) in &text {
                            admitted.insert(chunk.chunk_id.as_str(), (*chunk, true, false));
                        }
                        for (chunk, _) in &vectors {
                            admitted
                                .entry(chunk.chunk_id.as_str())
                                .or_insert((*chunk, false, false))
                                .2 = true;
                        }
                        let union_count = admitted.len();
                        let max_text = admitted
                            .values()
                            .map(|(chunk, _, _)| {
                                if has_text {
                                    text_score(chunk.ordinal).unwrap_or(0.0)
                                } else {
                                    0.0
                                }
                            })
                            .fold(0.0, f64::max);
                        let mut expected = admitted
                            .iter()
                            .map(|(&id, (chunk, _, _))| {
                                let text = if has_text {
                                    text_score(chunk.ordinal).unwrap_or(0.0)
                                } else {
                                    0.0
                                };
                                let vector = if has_vector {
                                    vector_score(chunk.ordinal)
                                } else {
                                    None
                                };
                                let bm25 = if max_text > 0.0 { text / max_text } else { 0.0 };
                                let cosine = vector.map(|raw| (raw + 1.0) / 2.0);
                                ExpectedScore {
                                    id,
                                    text_raw: has_text.then_some(text),
                                    vector_raw: vector,
                                    text_norm: bm25,
                                    vector_norm: cosine,
                                    final_score: match (has_text, has_vector) {
                                        (true, true) => 0.7 * bm25 + 0.3 * cosine.unwrap_or(0.0),
                                        (true, false) => bm25,
                                        (false, true) => cosine.unwrap_or(0.0),
                                        _ => unreachable!(),
                                    },
                                }
                            })
                            .collect::<Vec<_>>();
                        expected.sort_by(|a, b| {
                            b.final_score
                                .total_cmp(&a.final_score)
                                .then_with(|| a.id.cmp(b.id))
                        });
                        expected.truncate(limit);
                        let fused = expected.len();
                        expected.truncate(top_k);
                        let mut builder = collection.search();
                        if has_text {
                            builder = builder.text("needle");
                        }
                        if has_vector {
                            builder = builder
                                .query_embedding(QueryEmbeddingInput::from_vector(vec![1.0, 0.0]));
                        }
                        let actual = builder
                            .filter(if filtered {
                                parse_json(r#"{"chunk.eligible":true}"#).unwrap()
                            } else {
                                JsonValue::object()
                            })
                            .weights(SearchWeights::new(0.7, 0.3, 0.0).unwrap())
                            .resource_profile(profile)
                            .top_k(top_k)
                            .execute()
                            .unwrap();
                        assert_eq!(actual.query_context.text_present, has_text);
                        assert_eq!(actual.query_context.embedding_present, has_vector);
                        assert_eq!(actual.query_context.index_source, source);
                        assert_eq!(actual.query_context.candidate_limits.bm25, limit);
                        assert_eq!(actual.query_context.candidate_limits.vector, limit);
                        assert_eq!(actual.query_context.candidate_limits.fusion, limit);
                        assert_eq!(
                            actual.query_context.execution_stats.vector_scan_count,
                            total_scans
                        );
                        assert_eq!(
                            actual.query_context.execution_stats.distance_ops,
                            total_distance_vectors * 2
                        );
                        assert_eq!(
                            actual.query_context.execution_stats.fusion_candidate_count,
                            fused
                        );
                        assert_eq!(actual.results.len(), expected.len());
                        let dto = actual.to_json();
                        let counts = dto.get("candidate_counts").unwrap();
                        assert_eq!(rank(counts, "bm25"), Some(text_count));
                        assert_eq!(rank(counts, "vector"), Some(vector_count));
                        assert_eq!(rank(counts, "fusion"), Some(fused));

                        for (rank, (hit, expected)) in
                            actual.results.iter().zip(&expected).enumerate()
                        {
                            assert_eq!(
                            hit.chunk_id, expected.id,
                            "{source} {profile:?} k={top_k} filter={filtered} text={has_text} vector={has_vector}"
                        );
                            assert_eq!(hit.rank, rank + 1);
                            let normalization = hit.explain.get("normalization").unwrap();
                            assert_eq!(
                                self::rank(normalization, "candidate_count"),
                                Some(union_count)
                            );
                            if has_text {
                                assert!(
                                    (normalization.get("bm25_max").unwrap().as_f64().unwrap()
                                        - max_text)
                                        .abs()
                                        < 1e-12
                                );
                            } else {
                                assert_eq!(normalization.get("bm25_max"), Some(&JsonValue::Null));
                            }
                            let (_, b, v) = admitted[expected.id];
                            let source = match (b, v) {
                                (true, true) => r#"["bm25","vector"]"#,
                                (true, false) => r#"["bm25"]"#,
                                (false, true) => r#"["vector"]"#,
                                _ => panic!("U without recall source"),
                            };
                            assert_eq!(
                                hit.explain
                                    .get("candidate_sources")
                                    .unwrap()
                                    .to_compact_string(),
                                source
                            );

                            for (actual, wanted) in [
                                (hit.bm25_score, expected.text_raw),
                                (hit.vector_score, expected.vector_raw),
                                (hit.normalized_scores.vector_norm, expected.vector_norm),
                            ] {
                                match (actual, wanted) {
                                    (Some(a), Some(b)) => assert!((a - b).abs() < 1e-12),
                                    _ => assert_eq!(actual, wanted),
                                }
                            }
                            assert!(
                                (hit.normalized_scores.bm25_norm - expected.text_norm).abs()
                                    < 1e-12
                            );
                            assert!((hit.final_score - expected.final_score).abs() < 1e-12);
                            assert_eq!(hit.metadata_boost, 0.0);
                            let ordinal = hit
                                .chunk_external_id
                                .strip_prefix("chunk-")
                                .unwrap()
                                .parse()
                                .unwrap();
                            saw_missing_embedding |= vector(ordinal).is_none();
                            saw_vector_only |= frequency(ordinal) == 0;
                        }
                    }
                }
            }
        }
        drop(collection);
        db.close().unwrap();
        assert!(saw_missing_embedding && saw_vector_only, "{source}");
        assert_eq!(fs::read(&fixture.0).unwrap(), before);
    }
}

#[test]
fn normal_and_diagnostic_scoring_agree_across_sources_and_query_options() {
    let fixture = Fixture::new("matrix");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    for (index, metric) in [
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
        DistanceMetric::Euclidean,
    ]
    .into_iter()
    .enumerate()
    {
        let collection = db
            .create_collection(collection_config(format!("c{index}"), metric))
            .unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: parse_json(r#"{"owner":"reference"}"#).unwrap(),
                chunks: (0..12)
                    .map(|i| ChunkInput {
                        chunk_external_id: format!("chunk-{i}"),
                        text: if i % 3 == 0 {
                            "needle needle storage"
                        } else {
                            "needle index"
                        }
                        .into(),
                        ordinal: i,
                        metadata: parse_json(&format!(r#"{{"group":{}}}"#, i % 2)).unwrap(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(
                (0..11)
                    .map(|i| EmbeddingInput {
                        chunk_external_id: format!("chunk-{i}"),
                        model_name: "fixture".into(),
                        dim: 2,
                        vector: vec![(i % 3 + 1) as f32, (i % 4 + 1) as f32],
                    })
                    .collect(),
            )
            .unwrap();
        let deleted = collection.get_chunk_by_external_id("chunk-2").unwrap();
        collection.delete_chunk(&deleted.chunk_id).unwrap();
    }
    let critical_end = db
        .check(CheckOptions::default())
        .unwrap()
        .critical_end_offset;
    drop(db);
    let boost = MetadataBoostConfig::parse(parse_json(r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"chunk.group":1},"boost":0.75},{"when":{"doc.owner":"reference"},"boost":0.125}]}"#).unwrap()).unwrap();
    for source in ["persistent_checkpoint", "transient_rebuild"] {
        if source == "transient_rebuild" {
            fs::OpenOptions::new()
                .write(true)
                .open(&fixture.0)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
        }
        let db = Database::open_with_options(
            &fixture.0,
            DatabaseOpenOptions {
                read_only: true,
                ..Default::default()
            },
        )
        .unwrap();
        for seed in 0..100 {
            let name = format!("c{}", seed % 3);
            let collection = db.collection(&name).unwrap();
            let text =
                (seed / 3 % 3 != 1).then_some(if seed % 5 == 0 { "storage" } else { "needle" });
            let vector =
                (seed / 3 % 3 != 0).then(|| QueryEmbeddingInput::from_vector(vec![1.0, 2.0]));
            let filter = parse_json(match seed / 9 % 4 {
                0 => "{}",
                1 => r#"{"chunk.group":0}"#,
                2 => r#"{"chunk.group":1,"doc.owner":"reference"}"#,
                _ => r#"{"chunk.group":99}"#,
            })
            .unwrap();
            let enabled = seed % 2 == 0;
            let profile = [
                ResourceProfile::Balanced,
                ResourceProfile::Constrained,
                ResourceProfile::Expanded,
            ][seed / 4 % 3];
            let weights = SearchWeights::new(0.25, 0.5, 0.25).unwrap();
            let top_k = seed % 8 + 1;
            let mut query = collection
                .search()
                .filter(filter.clone())
                .weights(weights)
                .metadata_boost(boost.clone())
                .metadata_boost_enabled(enabled)
                .resource_profile(profile)
                .top_k(12);
            if let Some(text) = text {
                query = query.text(text);
            }
            if let Some(vector) = &vector {
                query = query.query_embedding(vector.clone());
            }
            let limited = query.clone().top_k(top_k).execute().unwrap();
            let all = query.execute().unwrap();
            assert_eq!(all.query_context.index_source, source);
            assert_eq!(limited.results.len(), all.results.len().min(top_k));
            for (limited, full) in limited.results.iter().zip(&all.results) {
                assert_eq!(limited.chunk_id, full.chunk_id);
                for field in ["matched_terms", "matched_term_details", "bm25"] {
                    assert_eq!(limited.explain.get(field), full.explain.get(field));
                }
            }
            for hit in &all.results {
                let ordinal: usize = hit.chunk_external_id[6..].parse().unwrap();
                let matched_term = text.filter(|term| *term != "storage" || ordinal % 3 == 0);
                let mut expected_sources = Vec::new();
                if matched_term.is_some() {
                    expected_sources.push(JsonValue::String("bm25".into()));
                }
                if vector.is_some() && hit.chunk_external_id != "chunk-11" {
                    expected_sources.push(JsonValue::String("vector".into()));
                }
                assert_eq!(
                    hit.explain.get("candidate_sources"),
                    Some(&JsonValue::Array(expected_sources)),
                    "seed={seed}, source={source}, chunk={}",
                    hit.chunk_external_id
                );
                let matched = hit.explain.get("matched_terms").unwrap();
                let compact = hit
                    .explain
                    .get("matched_term_details")
                    .unwrap()
                    .as_array()
                    .unwrap();
                let detailed = hit
                    .explain
                    .get("bm25")
                    .unwrap()
                    .get("terms")
                    .unwrap()
                    .as_array()
                    .unwrap();
                if let Some(term) = matched_term {
                    assert_eq!(
                        matched,
                        &JsonValue::Array(vec![JsonValue::String(term.into())])
                    );
                    assert_eq!(compact.len(), 1);
                    assert_eq!(detailed.len(), 1);
                    let (tf, df, length) = if term == "storage" {
                        (1.0, 4.0, 3.0)
                    } else if ordinal % 3 == 0 {
                        (2.0, 11.0, 3.0)
                    } else {
                        (1.0, 11.0, 2.0)
                    };
                    // Four three-token chunks and seven two-token chunks remain after deletion.
                    let avg_len = 26.0 / 11.0;
                    let raw = (12.0_f64 / (df + 0.5)).ln() * tf * 2.2
                        / (tf + 1.2 * (0.25 + 0.75 * length / avg_len));
                    assert_eq!(detailed[0].get("term").unwrap().as_str(), Some(term));
                    assert_eq!(detailed[0].get("tf").unwrap().as_f64(), Some(tf));
                    assert_eq!(detailed[0].get("df").unwrap().as_f64(), Some(df));
                    assert_eq!(
                        detailed[0].get("chunk_length").unwrap().as_f64(),
                        Some(length)
                    );
                    assert_eq!(
                        detailed[0].get("avg_chunk_length").unwrap().as_f64(),
                        Some(avg_len)
                    );
                    assert_eq!(compact[0].get("term").unwrap().as_str(), Some(term));
                    assert_eq!(compact[0].get("frequency").unwrap().as_f64(), Some(tf));
                    for actual in [
                        detailed[0]
                            .get("raw_contribution")
                            .unwrap()
                            .as_f64()
                            .unwrap(),
                        compact[0]
                            .get("bm25_contribution")
                            .unwrap()
                            .as_f64()
                            .unwrap(),
                        hit.bm25_score.unwrap(),
                    ] {
                        assert!((actual - raw).abs() < 1e-12, "seed={seed}, source={source}");
                    }
                } else {
                    assert_eq!(matched, &JsonValue::Array(Vec::new()));
                    assert!(compact.is_empty());
                    assert!(detailed.is_empty());
                }
                let weights = hit.explain.get("weights").unwrap();
                let effective = weights.get("effective").unwrap();
                for field in ["bm25", "vector", "metadata"] {
                    assert_eq!(
                        weights.get(field),
                        effective.get(field),
                        "seed={seed}, source={source}"
                    );
                }
                let score = effective.get("bm25").unwrap().as_f64().unwrap()
                    * hit.normalized_scores.bm25_norm
                    + effective.get("vector").unwrap().as_f64().unwrap()
                        * hit.normalized_scores.vector_norm.unwrap_or(0.0)
                    + effective.get("metadata").unwrap().as_f64().unwrap() * hit.metadata_boost;
                assert!(
                    (score - hit.final_score).abs() < 1e-12,
                    "seed={seed}, source={source}"
                );
            }
            assert!(all
                .results
                .iter()
                .all(|result| result.chunk_external_id != "chunk-2"));
            let expected = format!("chunk-{}", seed / 5 % 12);
            let expected_rank = all
                .results
                .iter()
                .find(|result| result.chunk_external_id == expected)
                .map(|result| result.rank);
            let diagnosis = db
                .why_not_debug_with_metadata_boost(
                    &name,
                    &expected,
                    text.map(str::to_owned),
                    vector,
                    filter,
                    Some(weights),
                    enabled,
                    Some(boost.clone()),
                    top_k,
                    Some(profile),
                )
                .unwrap();
            assert_eq!(diagnosis.query_context.index_source, source);
            assert_eq!(diagnosis.query_context.metadata_boost_enabled, enabled);
            let result = &diagnosis.result;
            assert_eq!(
                rank(result, "returned_rank"),
                expected_rank.filter(|rank| *rank <= top_k),
                "seed={seed}, source={source}"
            );
            assert_eq!(
                rank(result.get("candidate_ranks").unwrap(), "fusion_rank"),
                expected_rank,
                "seed={seed}, source={source}"
            );
            assert!(diagnosis.debug_full_scan);
        }
    }
}

#[test]
fn diagnostic_zero_weight_branch_does_not_blame_a_limit_after_union_admission() {
    for zero_bm25 in [true, false] {
        let fixture = Fixture::new(if zero_bm25 {
            "zero-bm25-limit"
        } else {
            "zero-vector-limit"
        });
        let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
            .unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..51)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("chunk-{ordinal}"),
                        ordinal,
                        text: if ordinal == 0 || (zero_bm25 && ordinal != 50) {
                            "needle needle needle hay"
                        } else if !zero_bm25 && ordinal == 50 {
                            "needle needle hay"
                        } else {
                            "needle hay"
                        }
                        .into(),
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(
                (0..51)
                    .map(|ordinal| EmbeddingInput {
                        chunk_external_id: format!("chunk-{ordinal}"),
                        model_name: "test".into(),
                        dim: 2,
                        vector: if ordinal == 0 || (!zero_bm25 && ordinal != 50) {
                            vec![1.0, 0.0]
                        } else if zero_bm25 && ordinal == 50 {
                            vec![1.0, 1.0]
                        } else {
                            vec![0.0, 1.0]
                        },
                    })
                    .collect(),
            )
            .unwrap();
        let critical_end = db
            .check(CheckOptions::default())
            .unwrap()
            .critical_end_offset;
        drop(collection);
        db.close().unwrap();
        let weights = if zero_bm25 {
            SearchWeights::new(0.0, 1.0, 0.0).unwrap()
        } else {
            SearchWeights::new(1.0, 0.0, 0.0).unwrap()
        };
        for source in ["persistent_checkpoint", "transient_rebuild"] {
            if source == "transient_rebuild" {
                fs::OpenOptions::new()
                    .write(true)
                    .open(&fixture.0)
                    .unwrap()
                    .set_len(critical_end)
                    .unwrap();
            }
            let before = fs::read(&fixture.0).unwrap();
            let db = Database::open_read_only(&fixture.0).unwrap();
            let query = QueryEmbeddingInput {
                model_name: Some("test".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            };
            let actual = db
                .collection("notes")
                .unwrap()
                .search()
                .text("needle")
                .query_embedding(query.clone())
                .weights(weights)
                .top_k(1)
                .resource_profile(ResourceProfile::Constrained)
                .execute()
                .unwrap();
            assert_eq!(actual.query_context.index_source, source);
            assert_eq!(actual.results[0].chunk_external_id, "chunk-0");
            let diagnosis = db
                .why_not_debug(
                    "notes",
                    "chunk-50",
                    Some("needle".into()),
                    Some(query),
                    JsonValue::object(),
                    Some(weights),
                    false,
                    1,
                    Some(ResourceProfile::Constrained),
                )
                .unwrap();
            let result = &diagnosis.result;
            let ranks = result.get("candidate_ranks").unwrap();
            assert_eq!(
                rank(
                    ranks,
                    if zero_bm25 {
                        "bm25_rank"
                    } else {
                        "vector_rank"
                    }
                ),
                Some(51)
            );
            assert_eq!(
                rank(
                    ranks,
                    if zero_bm25 {
                        "vector_rank"
                    } else {
                        "bm25_rank"
                    }
                ),
                Some(2)
            );
            assert_eq!(rank(ranks, "fusion_rank"), Some(2));
            assert_eq!(rank(result, "returned_rank"), None);
            let analysis = result.get("limit_analysis").unwrap();
            assert_eq!(
                analysis
                    .get("missed_due_to_candidate_limit")
                    .unwrap()
                    .as_bool(),
                Some(false)
            );
            for field in [
                "limiting_stage",
                "limiting_parameter",
                "effective_candidate_limit",
            ] {
                assert!(analysis.get(field).unwrap().is_null(), "{field}");
            }
            let codes = result.get("diagnosis_codes").unwrap().as_array().unwrap();
            assert!(codes
                .iter()
                .any(|code| code.as_str() == Some("rank_below_top_k")));
            assert!(!codes
                .iter()
                .any(|code| code.as_str() == Some("missed_due_to_candidate_limit")));
            db.close().unwrap();
            assert_eq!(fs::read(&fixture.0).unwrap(), before);
        }
    }
}

#[test]
fn diagnostic_ranks_extend_beyond_normal_candidate_limits() {
    let fixture = Fixture::new("full-ranks");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
        .unwrap();
    insert_tied_chunks(&collection);
    let mut ids = collection
        .list_chunks_by_document_external_id("doc")
        .unwrap();
    ids.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
    let diagnosis = db
        .why_not_debug(
            "notes",
            &ids[60].chunk_external_id,
            Some("needle".into()),
            None,
            JsonValue::object(),
            None,
            false,
            5,
            Some(ResourceProfile::Constrained),
        )
        .unwrap();
    let ranks = diagnosis.result.get("candidate_ranks").unwrap();
    assert_eq!(rank(ranks, "bm25_rank"), Some(61));
    assert_eq!(rank(ranks, "fusion_rank"), None);
    assert_eq!(rank(&diagnosis.result, "returned_rank"), None);
    assert_eq!(
        diagnosis
            .result
            .get("limit_analysis")
            .unwrap()
            .get("missed_due_to_candidate_limit")
            .unwrap()
            .as_bool(),
        Some(false)
    );
}

#[test]
fn diagnostic_candidate_limit_uses_bm25_rank_before_postfilter() {
    let fixture = Fixture::new("postfilter-limit");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
        .unwrap();
    insert_tied_chunks(&collection);
    let mut chunks = collection
        .list_chunks_by_document_external_id("doc")
        .unwrap();
    chunks.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
    for profile in [
        ResourceProfile::Constrained,
        ResourceProfile::Balanced,
        ResourceProfile::Expanded,
    ] {
        for (position, chunk) in chunks.iter().enumerate() {
            let limited = position >= profile.params().bm25_candidates;
            let filter = parse_json(&format!(r#"{{"chunk.index":{}}}"#, chunk.ordinal)).unwrap();
            let actual = collection
                .search()
                .text("needle")
                .filter(filter.clone())
                .resource_profile(profile)
                .top_k(5)
                .execute()
                .unwrap();
            assert_eq!(actual.results.is_empty(), limited);
            let diagnosis = db
                .why_not_debug(
                    "notes",
                    &chunk.chunk_external_id,
                    Some("needle".into()),
                    None,
                    filter,
                    None,
                    false,
                    5,
                    Some(profile),
                )
                .unwrap();
            let result = &diagnosis.result;
            let ranks = result.get("candidate_ranks").unwrap();
            assert_eq!(rank(ranks, "bm25_rank"), Some(1));
            assert_eq!(rank(ranks, "fusion_rank"), (!limited).then_some(1));
            assert_eq!(rank(result, "returned_rank"), (!limited).then_some(1));
            let analysis = result.get("limit_analysis").unwrap();
            assert_eq!(
                analysis
                    .get("missed_due_to_candidate_limit")
                    .unwrap()
                    .as_bool(),
                Some(limited),
                "position={position}, profile={profile:?}"
            );
            assert_eq!(
                analysis.get("limiting_stage").unwrap().as_str(),
                limited.then_some("bm25")
            );
            assert_eq!(
                rank(analysis, "effective_candidate_limit"),
                limited.then_some(profile.params().bm25_candidates)
            );
        }
    }
}

#[test]
fn diagnostic_fusion_limit_uses_profile_candidates_not_full_scan_rank() {
    let fixture = Fixture::new("fusion-limit");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
        .unwrap();
    insert_tied_chunks(&collection);
    collection
        .insert_document(DocumentInput {
            external_id: "target".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "target".into(),
                text: "unrelated".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
    let mut embeddings = (0..75)
        .map(|i| EmbeddingInput {
            chunk_external_id: format!("chunk-{i}"),
            model_name: "fixture".into(),
            dim: 2,
            vector: vec![0.6, 0.8],
        })
        .collect::<Vec<_>>();
    embeddings.push(EmbeddingInput {
        chunk_external_id: "target".into(),
        model_name: "fixture".into(),
        dim: 2,
        vector: vec![1.0, 0.0],
    });
    collection.put_embeddings(embeddings).unwrap();
    let critical_end = db
        .check(CheckOptions::default())
        .unwrap()
        .critical_end_offset;
    drop(collection);
    drop(db);
    for source in ["persistent_checkpoint", "transient_rebuild"] {
        if source == "transient_rebuild" {
            fs::OpenOptions::new()
                .write(true)
                .open(&fixture.0)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
        }
        let db = Database::open_with_options(
            &fixture.0,
            DatabaseOpenOptions {
                read_only: true,
                ..Default::default()
            },
        )
        .unwrap();
        let collection = db.collection("notes").unwrap();
        for profile in [
            ResourceProfile::Constrained,
            ResourceProfile::Balanced,
            ResourceProfile::Expanded,
        ] {
            for top_k in [5, 50] {
                let weights = SearchWeights::new(0.4, 0.6, 0.0).unwrap();
                let vector = QueryEmbeddingInput::from_vector(vec![1.0, 0.0]);
                let results = collection
                    .search()
                    .text("needle")
                    .query_embedding(vector.clone())
                    .weights(weights)
                    .resource_profile(profile)
                    .top_k(top_k)
                    .execute()
                    .unwrap();
                let expected_rank = None;
                assert_eq!(
                    results
                        .results
                        .iter()
                        .find(|hit| hit.chunk_external_id == "target")
                        .map(|hit| hit.rank),
                    expected_rank
                );
                let diagnosis = db
                    .why_not_debug(
                        "notes",
                        "target",
                        Some("needle".into()),
                        Some(vector),
                        JsonValue::object(),
                        Some(weights),
                        false,
                        top_k,
                        Some(profile),
                    )
                    .unwrap();
                assert_eq!(diagnosis.query_context.index_source, source);
                assert_eq!(rank(&diagnosis.result, "returned_rank"), expected_rank);
                let ranks = diagnosis.result.get("candidate_ranks").unwrap();
                assert_eq!(rank(ranks, "bm25_rank"), None);
                assert_eq!(rank(ranks, "vector_rank"), Some(1));
                let actual_fusion_rank = if profile == ResourceProfile::Constrained {
                    51
                } else {
                    76
                };
                assert_eq!(rank(ranks, "fusion_rank"), Some(actual_fusion_rank));
                let analysis = diagnosis.result.get("limit_analysis").unwrap();
                // Full scores rank target below top_k in every profile. Resource-first
                // additionally truncates it at fusion; removing that limit still cannot return it.
                assert_eq!(
                    analysis
                        .get("missed_due_to_candidate_limit")
                        .and_then(JsonValue::as_bool),
                    Some(false)
                );
                let truncated = profile == ResourceProfile::Constrained;
                assert_eq!(
                    analysis.get("limiting_stage").unwrap().as_str(),
                    truncated.then_some("fusion")
                );
                assert_eq!(
                    analysis.get("limiting_parameter").unwrap().as_str(),
                    truncated.then_some("fusion_candidates")
                );
                assert_eq!(
                    rank(analysis, "effective_candidate_limit"),
                    truncated.then_some(50)
                );
            }
        }
    }
}

#[test]
fn metadata_type_contract_survives_reopen_and_index_fallback() {
    let fixture = Fixture::new("metadata-types");
    let values = [
        Some("null"),
        Some("false"),
        Some("true"),
        Some("0"),
        Some("1"),
        Some(r#""1""#),
        Some(r#""A""#),
        Some(r#""a""#),
        Some("[]"),
        Some(r#"[1,"a",null]"#),
        Some(r#"{"nested":1}"#),
        None,
    ];
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("types".into(), DistanceMetric::Cosine))
        .unwrap();
    for index in 0..values.len() {
        let metadata = |value: Option<&str>| {
            value.map_or_else(JsonValue::object, |value| {
                parse_json(&format!(r#"{{"value":{value}}}"#)).unwrap()
            })
        };
        let mut chunk_metadata = metadata(values[values.len() - 1 - index]);
        chunk_metadata
            .as_object_mut()
            .unwrap()
            .insert("keep".into(), JsonValue::Bool(index % 2 == 0));
        collection
            .insert_document(DocumentInput {
                external_id: format!("doc-{index}"),
                title: None,
                source_uri: None,
                metadata: metadata(values[index]),
                chunks: vec![ChunkInput {
                    chunk_external_id: format!("chunk-{index}"),
                    text: "needle".into(),
                    ordinal: index,
                    metadata: chunk_metadata,
                    source_span: None,
                }],
            })
            .unwrap();
    }
    collection
        .put_embeddings(
            (0..values.len())
                .map(|index| EmbeddingInput {
                    chunk_external_id: format!("chunk-{index}"),
                    model_name: "types".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .collect(),
        )
        .unwrap();
    let critical_end = db
        .check(CheckOptions::default())
        .unwrap()
        .critical_end_offset;
    drop(collection);
    drop(db);

    // Each expected set refers to the literal table above, independently of FilterExpr.
    let cases: [(&str, &[usize]); 25] = [
        ("null", &[0]),
        ("false", &[1]),
        ("true", &[2]),
        (r#"{"$eq":0}"#, &[3]),
        (r#"{"$eq":1.0}"#, &[4]),
        (r#""1""#, &[5]),
        (r#""A""#, &[6]),
        (r#""a""#, &[7]),
        (r#"{"$eq":[]}"#, &[8]),
        (r#"{"$eq":[1,"a",null]}"#, &[9]),
        (r#"{"$ne":1}"#, &[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]),
        (r#"{"$gt":0}"#, &[4]),
        (r#"{"$gte":0}"#, &[3, 4]),
        (r#"{"$lt":1}"#, &[3]),
        (r#"{"$lte":1}"#, &[3, 4]),
        (r#"{"$gt":"A"}"#, &[7]),
        (r#"{"$gte":"A"}"#, &[6, 7]),
        (r#"{"$lt":"A"}"#, &[5]),
        (r#"{"$lte":"A"}"#, &[5, 6]),
        (r#"{"$in":[null,1,"a"]}"#, &[0, 4, 7]),
        (r#"{"$contains":1}"#, &[9]),
        (r#"{"$contains":"a"}"#, &[9]),
        (r#"{"$contains":null}"#, &[9]),
        (r#"{"$contains":true}"#, &[]),
        (r#"{"$in":[]}"#, &[]),
    ];
    for source in ["persistent_checkpoint", "transient_rebuild"] {
        if source == "transient_rebuild" {
            fs::OpenOptions::new()
                .write(true)
                .open(&fixture.0)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
        }
        let original = fs::read(&fixture.0).unwrap();
        let db = Database::open_read_only(&fixture.0).unwrap();
        let collection = db.collection("types").unwrap();
        for seed in 0..100 {
            let (condition, expected_values) = cases[seed % cases.len()];
            let document_scope = seed / cases.len() % 2 == 0;
            let scope = if document_scope { "doc" } else { "chunk" };
            let with_and = seed >= 50;
            let expression = format!(r#"{{"{scope}.value":{condition}}}"#);
            let filter = parse_json(&if with_and {
                format!(r#"{{"$and":[{expression},{{"chunk.keep":true}}]}}"#)
            } else {
                expression
            })
            .unwrap();
            let mut expected = expected_values
                .iter()
                .map(|index| {
                    if document_scope {
                        *index
                    } else {
                        values.len() - 1 - index
                    }
                })
                .filter(|index| !with_and || index % 2 == 0)
                .map(|index| format!("chunk-{index}"))
                .collect::<Vec<_>>();
            expected.sort();
            for mode in 0..3 {
                let mut query = collection
                    .search()
                    .filter(filter.clone())
                    .top_k(values.len());
                if mode != 1 {
                    query = query.text("needle");
                }
                if mode != 0 {
                    query = query.query_embedding(QueryEmbeddingInput::from_vector(vec![1.0, 0.0]));
                }
                let result = query.execute().unwrap();
                assert_eq!(result.query_context.index_source, source);
                let mut actual = result
                    .results
                    .iter()
                    .map(|hit| hit.chunk_external_id.clone())
                    .collect::<Vec<_>>();
                actual.sort();
                assert_eq!(
                    actual, expected,
                    "seed={seed}, mode={mode}, source={source}"
                );
                for hit in &result.results {
                    assert_eq!(
                        hit.explain
                            .get("filter")
                            .unwrap()
                            .get("matched")
                            .unwrap()
                            .as_bool(),
                        Some(true)
                    );
                }
            }
            let selected = format!("chunk-{}", seed % values.len());
            let diagnosis = db
                .why_not_debug(
                    "types",
                    &selected,
                    Some("needle".into()),
                    None,
                    filter.clone(),
                    None,
                    false,
                    values.len(),
                    None,
                )
                .unwrap();
            assert_eq!(
                diagnosis
                    .result
                    .get("filter_result")
                    .unwrap()
                    .get("matched")
                    .unwrap()
                    .as_bool(),
                Some(expected.contains(&selected))
            );

            let boost = MetadataBoostConfig::parse(parse_json(&format!(r#"{{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{{"when":{},"boost":0.75}}]}}"#, filter.to_compact_string())).unwrap()).unwrap();
            let boosted = collection
                .search()
                .text("needle")
                .metadata_boost(boost)
                .top_k(values.len())
                .execute()
                .unwrap();
            assert_eq!(boosted.results.len(), values.len());
            for hit in boosted.results {
                assert_eq!(
                    hit.metadata_boost,
                    if expected.contains(&hit.chunk_external_id) {
                        0.75
                    } else {
                        0.0
                    }
                );
            }
        }
        for invalid in [
            r#"{"value":1}"#,
            r#"{"doc.value.nested":1}"#,
            r#"{"doc.value":{"$eq":{"nested":1}}}"#,
            r#"{"doc.value":{"$eq":[[1]]}}"#,
            r#"{"doc.value":{"$in":[[]]}}"#,
            r#"{"doc.value":{"$contains":[]}}"#,
            r#"{"doc.value":{"$gte":true}}"#,
            r#"{"doc.value":{"$eq":1,"$ne":2}}"#,
            r#"{"doc.value":{"$unknown":1}}"#,
            r#"{"$and":{}}"#,
        ] {
            let error = collection
                .search()
                .text("needle")
                .filter(parse_json(invalid).unwrap())
                .execute()
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter, "{invalid}");
        }
        assert_eq!(fs::read(&fixture.0).unwrap(), original);
    }
}

#[test]
fn metadata_keys_never_alias_document_or_chunk_fields() {
    let fixture = Fixture::new("metadata-field-collisions");
    let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(collection_config("keys".into(), DistanceMetric::Cosine))
        .unwrap();
    for index in 0..3 {
        let metadata = match index {
            0 => parse_json(r#"{"external_id":"metadata","source_uri":"metadata","chunk_external_id":"metadata"}"#).unwrap(),
            1 => JsonValue::object(),
            _ => parse_json(r#"{"external_id":null,"source_uri":null,"chunk_external_id":null}"#).unwrap(),
        };
        collection
            .insert_document(DocumentInput {
                external_id: format!("fixed-doc-{index}"),
                title: None,
                source_uri: Some("fixed-uri".into()),
                metadata: metadata.clone(),
                chunks: vec![ChunkInput {
                    chunk_external_id: format!("fixed-chunk-{index}"),
                    text: "needle".into(),
                    ordinal: 0,
                    metadata,
                    source_span: None,
                }],
            })
            .unwrap();
    }
    drop(collection);
    drop(db);
    let original = fs::read(&fixture.0).unwrap();
    let db = Database::open_read_only(&fixture.0).unwrap();
    let collection = db.collection("keys").unwrap();
    for field in [
        "doc.external_id",
        "doc.source_uri",
        "chunk.chunk_external_id",
    ] {
        for (condition, expected) in [
            (r#""metadata""#, vec!["fixed-chunk-0"]),
            ("null", vec!["fixed-chunk-2"]),
            (
                r#"{"$ne":"metadata"}"#,
                vec!["fixed-chunk-1", "fixed-chunk-2"],
            ),
            (r#""fixed-doc-0""#, vec![]),
            (r#""fixed-uri""#, vec![]),
            (r#""fixed-chunk-0""#, vec![]),
        ] {
            let filter = parse_json(&format!(r#"{{"{field}":{condition}}}"#)).unwrap();
            let result = collection
                .search()
                .text("needle")
                .filter(filter.clone())
                .execute()
                .unwrap();
            let mut actual = result
                .results
                .iter()
                .map(|hit| hit.chunk_external_id.as_str())
                .collect::<Vec<_>>();
            actual.sort();
            assert_eq!(actual, expected, "{field}: {condition}");
            let boost = MetadataBoostConfig::parse(parse_json(&format!(r#"{{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{{"when":{},"boost":1}}]}}"#, filter.to_compact_string())).unwrap()).unwrap();
            let boosted = collection
                .search()
                .text("needle")
                .metadata_boost(boost)
                .execute()
                .unwrap();
            for hit in boosted.results {
                assert_eq!(
                    hit.metadata_boost,
                    f64::from(expected.contains(&hit.chunk_external_id.as_str())),
                    "{field}: {condition}"
                );
            }
            for index in 0..3 {
                let id = format!("fixed-chunk-{index}");
                let diagnosis = db
                    .why_not_debug(
                        "keys",
                        &id,
                        Some("needle".into()),
                        None,
                        filter.clone(),
                        None,
                        false,
                        10,
                        None,
                    )
                    .unwrap();
                assert_eq!(
                    diagnosis
                        .result
                        .get("filter_result")
                        .unwrap()
                        .get("matched")
                        .unwrap()
                        .as_bool(),
                    Some(expected.contains(&id.as_str())),
                    "{field}: {condition}"
                );
            }
        }
    }
    assert_eq!(fs::read(&fixture.0).unwrap(), original);
}

#[test]
fn complete_scores_use_collection_statistics_and_keep_vector_only_provenance() {
    for transient in [false, true] {
        let fixture = Fixture::new(if transient {
            "complete-filter-transient"
        } else {
            "complete-filter-indexed"
        });
        let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
            .unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..54)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("c-{ordinal}"),
                        ordinal,
                        text: match ordinal {
                            0..=50 => "needle ".repeat(10),
                            51 => format!("needle {}", "filler ".repeat(99)),
                            52 => format!("needle needle {}", "filler ".repeat(98)),
                            _ => "filler ".repeat(100),
                        },
                        metadata: parse_json(&format!(r#"{{"eligible":{}}}"#, ordinal >= 51))
                            .unwrap(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(
                (51..54)
                    .map(|ordinal| EmbeddingInput {
                        chunk_external_id: format!("c-{ordinal}"),
                        model_name: "fixture".into(),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    })
                    .collect(),
            )
            .unwrap();
        let critical_end = db
            .check(CheckOptions::default())
            .unwrap()
            .critical_end_offset;
        drop(collection);
        db.close().unwrap();
        if transient {
            fs::OpenOptions::new()
                .write(true)
                .open(&fixture.0)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
        }
        let before = fs::read(&fixture.0).unwrap();
        let db = Database::open_read_only(&fixture.0).unwrap();
        for text_weight in [0.0, 0.7] {
            let results = db
                .collection("notes")
                .unwrap()
                .search()
                .text("needle")
                .query_embedding(QueryEmbeddingInput::from_vector(vec![1.0, 0.0]))
                .filter(parse_json(r#"{"chunk.eligible":true}"#).unwrap())
                .resource_profile(ResourceProfile::Constrained)
                .top_k(3)
                .weights(SearchWeights::new(text_weight, 1.0 - text_weight, 0.0).unwrap())
                .execute()
                .unwrap();
            assert_eq!(results.results.len(), 3);
            let dto = results.to_json();
            let counts = dto.get("candidate_counts").unwrap();
            assert_eq!(rank(counts, "bm25"), Some(0));
            assert_eq!(rank(counts, "vector"), Some(3));
            assert_eq!(rank(counts, "fusion"), Some(3));
            let average = 810.0 / 54.0;
            let idf = ((54.0_f64 + 1.0) / (53.0 + 0.5)).ln();
            let raw = |tf: f64| {
                if tf == 0.0 {
                    0.0
                } else {
                    idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * 100.0 / average))
                }
            };
            for hit in &results.results {
                let tf = match hit.chunk_external_id.as_str() {
                    "c-51" => 1.0,
                    "c-52" => 2.0,
                    "c-53" => 0.0,
                    other => panic!("unexpected U member {other}"),
                };
                assert!((hit.bm25_score.unwrap() - raw(tf)).abs() < 1e-12);
                assert!((hit.normalized_scores.bm25_norm - raw(tf) / raw(2.0)).abs() < 1e-12);
                assert!(
                    (hit.final_score - (text_weight * raw(tf) / raw(2.0) + 1.0 - text_weight))
                        .abs()
                        < 1e-12
                );
                let explain = &hit.explain;
                assert_eq!(
                    explain
                        .get("candidate_sources")
                        .unwrap()
                        .to_compact_string(),
                    r#"["vector"]"#
                );
                let normalization = explain.get("normalization").unwrap();
                assert_eq!(rank(normalization, "candidate_count"), Some(3));
                assert!(
                    (normalization.get("bm25_max").unwrap().as_f64().unwrap() - raw(2.0)).abs()
                        < 1e-12
                );
                assert_eq!(
                    explain
                        .get("matched_terms")
                        .unwrap()
                        .as_array()
                        .unwrap()
                        .len(),
                    usize::from(tf != 0.0)
                );
            }
        }
        db.close().unwrap();
        assert_eq!(fs::read(&fixture.0).unwrap(), before);
    }
}

#[test]
fn diagnostic_checks_both_outside_windows_without_changing_ordinary_union() {
    for transient in [false, true] {
        let fixture = Fixture::new(if transient {
            "both-limits-transient"
        } else {
            "both-limits-indexed"
        });
        let db = Database::create(&fixture.0, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(collection_config("notes".into(), DistanceMetric::Cosine))
            .unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..102)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("c-{ordinal}"),
                        ordinal,
                        text: match ordinal {
                            0..=49 => "needle".into(),
                            50 => "needle filler".into(),
                            51 => format!("needle {}", "filler ".repeat(100)),
                            _ => "unrelated".into(),
                        },
                        metadata: parse_json(&format!(r#"{{"eligible":{}}}"#, ordinal >= 50))
                            .unwrap(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(
                (51..102)
                    .map(|ordinal| EmbeddingInput {
                        chunk_external_id: format!("c-{ordinal}"),
                        model_name: "fixture".into(),
                        dim: 2,
                        vector: if ordinal == 51 {
                            vec![0.8, 0.6]
                        } else {
                            vec![1.0, 0.0]
                        },
                    })
                    .collect(),
            )
            .unwrap();
        let critical_end = db
            .check(CheckOptions::default())
            .unwrap()
            .critical_end_offset;
        drop(collection);
        db.close().unwrap();
        if transient {
            fs::OpenOptions::new()
                .write(true)
                .open(&fixture.0)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
        }
        let original = fs::read(&fixture.0).unwrap();
        let db = Database::open_read_only(&fixture.0).unwrap();
        let filter = parse_json(r#"{"chunk.eligible":true}"#).unwrap();
        let vector = QueryEmbeddingInput::from_vector(vec![1.0, 0.0]);
        let weights = SearchWeights::new(0.5, 0.5, 0.0).unwrap();
        let ordinary = || {
            db.collection("notes")
                .unwrap()
                .search()
                .text("needle")
                .query_embedding(vector.clone())
                .filter(filter.clone())
                .weights(weights)
                .top_k(1)
                .resource_profile(ResourceProfile::Constrained)
                .execute()
                .unwrap()
        };
        let before = ordinary().to_json().to_compact_string();
        let average = 203.0 / 102.0;
        let idf = ((102.0_f64 + 1.0) / (52.0 + 0.5)).ln();
        let raw = |length: f64| idf * 2.2 / (1.0 + 1.2 * (0.25 + 0.75 * length / average));
        // Independent arithmetic proves BM25 expansion fails but vector expansion wins.
        assert!(0.5 * raw(101.0) / raw(2.0) + 0.5 * 0.9 < 0.5);
        assert!(0.5 * raw(101.0) / raw(101.0) + 0.5 * 0.9 > 0.5);
        let diagnosis = db
            .why_not_debug(
                "notes",
                "c-51",
                Some("needle".into()),
                Some(vector.clone()),
                filter.clone(),
                Some(weights),
                false,
                1,
                Some(ResourceProfile::Constrained),
            )
            .unwrap();
        let ranks = diagnosis.result.get("candidate_ranks").unwrap();
        assert_eq!(rank(ranks, "bm25_rank"), Some(2));
        assert_eq!(rank(ranks, "vector_rank"), Some(51));
        assert_eq!(rank(ranks, "fusion_rank"), None);
        let limits = diagnosis.result.get("limit_analysis").unwrap();
        assert_eq!(
            limits
                .get("missed_due_to_candidate_limit")
                .unwrap()
                .as_bool(),
            Some(true)
        );
        assert_eq!(
            limits.get("limiting_stage").unwrap().as_str(),
            Some("vector")
        );
        assert_eq!(
            limits.get("limiting_parameter").unwrap().as_str(),
            Some("vector_candidates")
        );
        assert_eq!(rank(limits, "effective_candidate_limit"), Some(50));
        let after = ordinary();
        assert_eq!(after.to_json().to_compact_string(), before);
        let hit = &after.results[0];
        assert_eq!(hit.final_score, 0.5);
        assert_eq!(hit.bm25_score, Some(0.0));
        assert_eq!(
            hit.explain
                .get("candidate_sources")
                .unwrap()
                .to_compact_string(),
            r#"["vector"]"#
        );
        assert_eq!(
            rank(hit.explain.get("normalization").unwrap(), "candidate_count"),
            Some(50)
        );
        assert_eq!(
            hit.explain
                .get("normalization")
                .unwrap()
                .get("bm25_max")
                .unwrap()
                .as_f64(),
            Some(0.0)
        );
        db.close().unwrap();
        assert_eq!(fs::read(&fixture.0).unwrap(), original);
    }
}
