use cofferdb::{
    parse_json, ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric,
    DocumentInput, EmbeddingInput, ErrorCode, JsonValue, MetadataBoostConfig, QueryEmbeddingInput,
    ResourceProfile, SearchWeights, VectorIndexConfig,
};
use std::{fs, path::PathBuf, process::Command};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// Inspect the original numeric token, allowing the existing pretty writer's whitespace.
fn assert_reported_top_k(raw: &str, expected: usize, minimum_occurrences: usize) {
    let mut seen = 0;
    for (offset, _) in raw.match_indices("\"top_k\"") {
        let suffix = raw[offset + "\"top_k\"".len()..]
            .trim_start()
            .strip_prefix(':')
            .unwrap()
            .trim_start();
        let end = suffix
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(suffix.len());
        assert_eq!(suffix[..end].parse::<usize>().unwrap(), expected);
        seen += 1;
    }
    assert!(
        seen >= minimum_occurrences,
        "top_k missing from original output: {raw}"
    );
}

#[test]
fn exact_top_k_range_agrees_across_api_and_real_cli_and_preserves_readonly_state() {
    let dir = std::env::temp_dir().join(format!("cofferdb-exact-top-k-{}", std::process::id()));
    assert!(!dir.exists(), "stale fixture: {}", dir.display());
    fs::create_dir(&dir).unwrap();
    let fixture = Fixture(dir);
    let path = fixture.0.join("db.coffer");
    let writer = Database::create(&path, DatabaseConfig::default()).unwrap();
    for name in ["one", "empty"] {
        let collection = writer
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: name.into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        if name == "one" {
            collection
                .insert_document(DocumentInput {
                    external_id: "doc".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: "chunk".into(),
                        text: "needle hay".into(),
                        ordinal: 0,
                        metadata: parse_json(r#"{"eligible":true}"#).unwrap(),
                        source_span: None,
                    }],
                })
                .unwrap();
            collection
                .put_embeddings(vec![EmbeddingInput {
                    chunk_external_id: "chunk".into(),
                    model_name: "m".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                }])
                .unwrap();
        }
    }
    writer.close().unwrap();
    let before = fs::read(&path).unwrap();
    let embedding_file = fixture.0.join("embedding.json");
    fs::write(
        &embedding_file,
        r#"{"model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
    )
    .unwrap();
    let maximum = if usize::BITS > 53 {
        9_007_199_254_740_991u64 as usize
    } else {
        usize::MAX
    };
    let db = Database::open_read_only(&path).unwrap();
    for name in ["one", "empty"] {
        let collection = db.collection(name).unwrap();
        for profile in [
            ResourceProfile::Balanced,
            ResourceProfile::Constrained,
            ResourceProfile::Expanded,
        ] {
            for k in [1, maximum - 1, maximum] {
                for (text, vector) in [(true, false), (false, true), (true, true)] {
                    let mut builder = collection.search().resource_profile(profile).top_k(k);
                    if text {
                        builder = builder.text("needle");
                    }
                    if vector {
                        builder = builder.query_embedding(QueryEmbeddingInput {
                            model_name: Some("m".into()),
                            dim: 2,
                            vector: vec![1.0, 0.0],
                        });
                    }
                    let expected = builder.execute().unwrap();
                    assert_eq!(expected.top_k, k);
                    assert_eq!(expected.query_context.top_k, k);
                    let limits = expected.query_context.candidate_limits;
                    let limit = k.max(profile.params().bm25_candidates);
                    assert_eq!(
                        (limits.bm25, limits.vector, limits.fusion),
                        (limit, limit, limit)
                    );
                    assert_eq!(expected.results.len(), usize::from(name == "one"));
                    for command in if text {
                        &["search"][..]
                    } else {
                        &["search", "vector-search"][..]
                    } {
                        let mut args = vec![command.to_string(), path.display().to_string()];
                        if text {
                            args.push("needle".into());
                        }
                        args.extend([
                            "--collection".into(),
                            name.into(),
                            "--profile".into(),
                            profile.as_cli_str().into(),
                            "--top-k".into(),
                            k.to_string(),
                            "--json".into(),
                        ]);
                        if vector {
                            args.extend([
                                "--embedding".into(),
                                embedding_file.display().to_string(),
                            ]);
                        }
                        let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
                            .args(&args)
                            .output()
                            .unwrap();
                        assert!(
                            output.status.success(),
                            "{args:?}: {}",
                            String::from_utf8_lossy(&output.stderr)
                        );
                        let raw = std::str::from_utf8(&output.stdout).unwrap();
                        assert_reported_top_k(raw, k, 2);
                        let envelope = parse_json(raw).unwrap();
                        let result = envelope.get("result").unwrap();
                        assert_eq!(
                            result,
                            &expected.to_json(),
                            "{name} {profile:?} {k} text={text} vector={vector} command={command}"
                        );
                        for hit in result.get("results").unwrap().as_array().unwrap() {
                            let effective =
                                hit.get("explain").unwrap().get("candidate_limits").unwrap();
                            for branch in ["bm25", "vector", "fusion"] {
                                assert_eq!(
                                    effective.get(branch).and_then(JsonValue::as_f64),
                                    Some(limit as f64)
                                );
                            }
                        }
                    }
                }
            }
        }
        let mut rejected = vec![0];
        if let Some(next) = maximum.checked_add(1) {
            rejected.extend([next, next + 1]);
        }
        for k in rejected {
            let error = collection
                .search()
                .text("needle")
                .top_k(k)
                .execute()
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidSearchParams);
            for command in ["search", "vector-search", "why-not"] {
                let mut args = vec![command.to_string(), path.display().to_string()];
                if command == "why-not" {
                    args.extend([
                        "--chunk-id".into(),
                        "chunk".into(),
                        "--query".into(),
                        "needle".into(),
                    ]);
                } else if command == "search" {
                    args.push("needle".into());
                } else {
                    args.extend(["--embedding".into(), embedding_file.display().to_string()]);
                }
                args.extend([
                    "--collection".into(),
                    name.into(),
                    "--top-k".into(),
                    k.to_string(),
                    "--json".into(),
                ]);
                let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
                    .args(&args)
                    .output()
                    .unwrap();
                assert_eq!(output.status.code(), Some(1), "{args:?}");
                let envelope = parse_json(std::str::from_utf8(&output.stderr).unwrap()).unwrap();
                assert!(output.stdout.is_empty());
                assert_eq!(
                    envelope
                        .get("error")
                        .unwrap()
                        .get("code")
                        .and_then(JsonValue::as_str),
                    Some("invalid_search_params"),
                    "{args:?}: {envelope:?}"
                );
            }
            assert_eq!(
                collection
                    .search()
                    .text("needle")
                    .top_k(1)
                    .execute()
                    .unwrap()
                    .results
                    .len(),
                usize::from(name == "one")
            );
        }
    }
    // Human output uses the same accepted integer without a JSON-only admission rule.
    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .args([
            "search",
            path.to_str().unwrap(),
            "needle",
            "--collection",
            "one",
            "--top-k",
            &maximum.to_string(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let raw = std::str::from_utf8(&output.stdout).unwrap();
    assert_reported_top_k(raw, maximum, 1);
    assert!(raw.contains(&format!("  top_k: {maximum}\n")));
    if let Some(next) = maximum.checked_add(1) {
        let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
            .args([
                "search",
                path.to_str().unwrap(),
                "needle",
                "--collection",
                "one",
                "--top-k",
                &next.to_string(),
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(std::str::from_utf8(&output.stderr)
            .unwrap()
            .starts_with("invalid_search_params:"));
    }
    let diagnosis = db
        .why_not_debug(
            "one",
            "chunk",
            Some("needle".into()),
            None,
            JsonValue::object(),
            None,
            false,
            maximum,
            Some(ResourceProfile::Balanced),
        )
        .unwrap();
    assert_eq!(diagnosis.query_context.top_k, maximum);
    let expected = diagnosis.to_json();
    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .args([
            "why-not",
            path.to_str().unwrap(),
            "--collection",
            "one",
            "--chunk-external-id",
            "chunk",
            "--query",
            "needle",
            "--top-k",
            &maximum.to_string(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let raw = std::str::from_utf8(&output.stdout).unwrap();
    assert_reported_top_k(raw, maximum, 1);
    let json = parse_json(raw).unwrap();
    assert_eq!(json.get("result"), Some(&expected));
    for branch in ["bm25", "vector", "fusion"] {
        assert_eq!(
            json.get("result")
                .unwrap()
                .get("query_context")
                .unwrap()
                .get("candidate_limits")
                .unwrap()
                .get(branch)
                .and_then(JsonValue::as_f64),
            Some(maximum as f64)
        );
    }
    // Queryless search retains its weights and metadata boost while expanding the window.
    let boost_json = r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"chunk.eligible":true},"boost":0.1}]}"#;
    let boosted = db
        .collection("one")
        .unwrap()
        .search()
        .query_embedding(QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        })
        .weights(SearchWeights::new(0.5, 0.3, 0.2).unwrap())
        .metadata_boost(MetadataBoostConfig::parse(parse_json(boost_json).unwrap()).unwrap())
        .resource_profile(ResourceProfile::Constrained)
        .top_k(600)
        .execute()
        .unwrap();
    assert!(!boosted.query_context.text_present);
    assert_eq!(boosted.query, None);
    assert_eq!(boosted.query_context.candidate_counts.bm25, 0);
    assert_eq!(boosted.results[0].bm25_score, None);
    assert!((boosted.query_context.weights.effective.vector - 0.6).abs() < 1e-12);
    assert!((boosted.query_context.weights.effective.metadata - 0.4).abs() < 1e-12);
    assert!((boosted.results[0].metadata_boost - 0.1).abs() < 1e-12);
    assert!((boosted.results[0].final_score - 0.64).abs() < 1e-12);
    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .args([
            "search",
            path.to_str().unwrap(),
            "--collection",
            "one",
            "--embedding",
            embedding_file.to_str().unwrap(),
            "--weights",
            r#"{"bm25":0.5,"vector":0.3,"metadata":0.2}"#,
            "--metadata-boost",
            boost_json,
            "--profile",
            "constrained",
            "--top-k",
            "600",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let raw = std::str::from_utf8(&output.stdout).unwrap();
    assert_reported_top_k(raw, 600, 2);
    assert_eq!(
        parse_json(raw).unwrap().get("result"),
        Some(&boosted.to_json())
    );
    let overflow = format!("{}0", usize::MAX);
    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .args([
            "search",
            path.to_str().unwrap(),
            "needle",
            "--collection",
            "one",
            "--top-k",
            &overflow,
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let json = parse_json(std::str::from_utf8(&output.stderr).unwrap()).unwrap();
    assert_eq!(
        json.get("error")
            .unwrap()
            .get("code")
            .and_then(JsonValue::as_str),
        Some("cli_usage")
    );
    db.close().unwrap();
    assert_eq!(fs::read(&path).unwrap(), before);
}
