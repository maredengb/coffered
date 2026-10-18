use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use cofferdb::{parse_json, Database, DatabaseConfig, JsonValue};

// Fixed wire bytes and independently computed SHA-256 answers, including trailing newlines.
const ASSETS: &[(&str, &str, &str)] = &[
    (
        "documents",
        concat!(
            r#"{"external_id":"doc","metadata":{"owner":"local"}}"#,
            "\n"
        ),
        "b8f5d67e92fb0d40391ec0f80eecea3e2e05260f2b09e497eb66900b928e4999",
    ),
    (
        "chunks",
        concat!(
            r#"{"document_external_id":"doc","chunk_external_id":"c0","ordinal":0,"text":"recovery log","metadata":{"lang":"rust"}}"#,
            "\n",
            r#"{"document_external_id":"doc","chunk_external_id":"c1","ordinal":1,"text":"metadata filter","metadata":{"lang":"python"}}"#,
            "\n"
        ),
        "b55b6cfdaa364c03f3d9109207a777078f65d835c472bd5429eb2141f8ddb87a",
    ),
    (
        "embeddings",
        concat!(
            r#"{"chunk_external_id":"c0","model_name":"m","dim":2,"vector":[1,0]}"#,
            "\n",
            r#"{"chunk_external_id":"c1","model_name":"m","dim":2,"vector":[0,1]}"#,
            "\n"
        ),
        "3de2cf2f99e1786e84507feaca20d169eb75a57dc146fa336b1f897abb9e0310",
    ),
    (
        "queries",
        concat!(
            r#"{"query_id":"hit","text":"recovery","top_k":2,"filter":{}}"#,
            "\n",
            r#"{"query_id":"empty","text":"recovery","top_k":2,"filter":{"chunk.lang":"none"}}"#,
            "\n"
        ),
        "4a97cf65f58238487b5f5f9486ded4cf8f36013037e1d881cb412ad1215f154c",
    ),
    (
        "qrels",
        concat!(
            r#"{"query_id":"hit","chunk_external_id":"c0","relevance":2}"#,
            "\n",
            r#"{"query_id":"empty","chunk_external_id":"c0","relevance":0}"#,
            "\n"
        ),
        "95ed81b71f4cbb1a0dfb41faf9e5133b7392e2fbdeb2b80643ef1f7384f040a1",
    ),
    (
        "query_embeddings",
        concat!(
            r#"{"query_id":"hit","model_name":"m","dim":2,"vector":[1,0]}"#,
            "\n",
            r#"{"query_id":"empty","model_name":"m","dim":2,"vector":[1,0]}"#,
            "\n"
        ),
        "971017c0e18cbd01e900334f3149cde5eb96d2436aa51550ba65d5813723a8e1",
    ),
    (
        "inline_queries",
        concat!(
            r#"{"query_id":"hit","text":"recovery","top_k":2,"filter":{},"embedding":[1,0]}"#,
            "\n",
            r#"{"query_id":"empty","text":"recovery","top_k":2,"filter":{"chunk.lang":"none"},"embedding":[1,0]}"#,
            "\n"
        ),
        "190775891426344776d28b7d21f5bc7f79f28d49223b6803f32a1da42688c89e",
    ),
];

struct Fixture(PathBuf);

impl Fixture {
    fn new(label: &str, source: &str) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "cofferdb-benchmark-cli-{label}-{source}-{}",
            std::process::id(),
        ));
        fs::create_dir(&directory).unwrap();
        let fixture = Self(directory);
        let mut manifest = parse_json(r#"{"schema_version":"0.1","dataset_id":"cli-contract","version":"1","chunking":{"chunk_size":256,"chunk_overlap":0}}"#).unwrap();
        let object = manifest.as_object_mut().unwrap();
        object.insert(
            "query_embedding_source".into(),
            JsonValue::String(source.into()),
        );
        if source != "none" {
            object.insert("embedding_model".into(), JsonValue::String("m".into()));
            object.insert(
                "query_embedding_model".into(),
                JsonValue::String("m".into()),
            );
            object.insert("query_embedding_dim".into(), JsonValue::Number(2.0));
        }
        let mut hashes = JsonValue::object();
        for &(name, bytes, hash) in ASSETS {
            let name = match name {
                "embeddings" if source == "none" => continue,
                "query_embeddings" if source != "sidecar" => continue,
                "queries" if source == "inline" => continue,
                "inline_queries" if source != "inline" => continue,
                "inline_queries" => "queries",
                name => name,
            };
            let filename = format!("{name}.jsonl");
            fs::write(fixture.0.join(&filename), bytes).unwrap();
            object.insert(name.into(), JsonValue::String(filename));
            hashes
                .as_object_mut()
                .unwrap()
                .insert(format!("{name}_sha256"), JsonValue::String(hash.into()));
        }
        object.insert("hashes".into(), hashes);
        fs::write(
            fixture.0.join("dataset_manifest.json"),
            manifest.to_compact_string(),
        )
        .unwrap();
        fixture
    }

    fn command(&self, baseline: bool) -> Command {
        self.command_with_profile(baseline, "balanced")
    }

    fn command_with_profile(&self, baseline: bool, profile: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
        command
            .current_dir(&self.0)
            .arg("benchmark")
            .arg("data.coffer")
            .arg("--dataset")
            .arg("dataset_manifest.json")
            .args([
                "--collection",
                "notes",
                "--profile",
                profile,
                "--durability",
                "fast",
                "--json",
            ])
            .arg("--result-file")
            .arg("result.json");
        if baseline {
            command.arg("--require-baseline");
        } else {
            command
                .env("COFFERDB_SQLITE3", self.0.join("missing-sqlite3"))
                .env_remove("COFFERDB_SQLITE_VEC_EXTENSION");
        }
        command
    }

    fn run(&self, baseline: bool) -> JsonValue {
        let output = self.command(baseline).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
        assert_eq!(envelope.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = envelope.get("result").unwrap().clone();
        let published =
            parse_json(&fs::read_to_string(self.0.join("result.json")).unwrap()).unwrap();
        assert_eq!(published, result);
        result
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[cfg(unix)]
#[test]
fn benchmark_commit_comes_from_build_not_runtime_repository_or_environment() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new("source-identity", "none");
    let bin = fixture.0.join("bin");
    fs::create_dir(&bin).unwrap();
    let git = bin.join("git");
    fs::write(&git, "#!/bin/sh\nprintf called > \"$GIT_PROBE\"\nprintf '%s\\n' 1111111111111111111111111111111111111111\n").unwrap();
    fs::set_permissions(&git, fs::Permissions::from_mode(0o755)).unwrap();
    let marker = fixture.0.join("git-was-called");
    let mut paths = vec![bin];
    paths.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    let output = fixture
        .command(false)
        .env("PATH", std::env::join_paths(paths).unwrap())
        .env("GIT_PROBE", &marker)
        .env(
            "COFFERDB_GIT_COMMIT",
            "2222222222222222222222222222222222222222",
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !marker.exists(),
        "runtime Git metadata is not the engine's build identity"
    );
    let json = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
    let expected = option_env!("COFFERDB_GIT_COMMIT")
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("unknown");
    assert_eq!(
        json.get("result")
            .unwrap()
            .get("git_commit")
            .and_then(JsonValue::as_str),
        Some(expected)
    );
}

fn verify_runs(runs: &JsonValue, vectors: bool) {
    let runs = runs.as_array().unwrap();
    assert_eq!(runs.len(), if vectors { 6 } else { 2 });
    let mut groups = std::collections::BTreeSet::new();
    for run in runs {
        let mode = run.get("query_mode").unwrap().as_str().unwrap();
        let cache = run.get("cache_state").unwrap().as_str().unwrap();
        assert!(groups.insert((mode, cache)));
        assert!(["cold", "warm"].contains(&cache));
        assert!(["text_bm25", "vector_exact", "hybrid"].contains(&mode));
        let rows = run.get("per_query").unwrap().as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].get("query_id").and_then(JsonValue::as_str),
            Some("hit")
        );
        assert_eq!(
            rows[0].get("ranked_chunk_ids").unwrap().as_array().unwrap()[0].as_str(),
            Some("c0")
        );
        assert_eq!(rows[0].get("hit").and_then(JsonValue::as_bool), Some(true));
        assert_eq!(
            rows[1].get("query_id").and_then(JsonValue::as_str),
            Some("empty")
        );
        assert!(rows[1]
            .get("ranked_chunk_ids")
            .unwrap()
            .as_array()
            .unwrap()
            .is_empty());
        assert_eq!(rows[1].get("hit").and_then(JsonValue::as_bool), Some(false));
        for row in rows {
            assert!(row.get("latency_ms").unwrap().as_f64().unwrap() >= 0.0);
        }
        for metric in ["recall_at_5", "recall_at_10", "mrr"] {
            assert_eq!(
                run.get("metrics").unwrap().get(metric).unwrap().as_f64(),
                Some(0.5)
            );
        }
    }
}

#[test]
fn benchmark_cli_preserves_all_embedding_modes_and_judged_empty_results() {
    for source in ["none", "inline", "sidecar"] {
        let fixture = Fixture::new("modes", source);
        let result = fixture.run(false);
        assert_eq!(
            result.get("baseline_status").and_then(JsonValue::as_str),
            Some("not_run")
        );
        assert_eq!(
            result
                .get("dataset")
                .unwrap()
                .get("query_embedding_source")
                .and_then(JsonValue::as_str),
            Some(source)
        );
        assert_eq!(
            result
                .get("import_context")
                .unwrap()
                .get("batch_size")
                .and_then(JsonValue::as_f64),
            Some(2.0)
        );
        verify_runs(result.get("runs").unwrap(), source != "none");
    }
}

#[test]
fn benchmark_cli_rejects_unsupported_profiles_with_otherwise_valid_inputs() {
    for valid in ["expanded", "balanced", "constrained"] {
        let fixture = Fixture::new(&format!("profile-{valid}"), "none");
        let inputs: Vec<_> = fs::read_dir(&fixture.0)
            .unwrap()
            .map(|entry| {
                let path = entry.unwrap().path();
                let bytes = fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect();
        let output = fixture.command_with_profile(false, valid).output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(0),
            "{valid}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
        assert_eq!(envelope.get("ok"), Some(&JsonValue::Bool(true)));
        assert_eq!(
            envelope
                .get("result")
                .unwrap()
                .get("profile")
                .and_then(JsonValue::as_str),
            Some(valid)
        );
        let database = fixture.0.join("data.coffer");
        let before = fs::read(&database).unwrap();
        let report = fixture.0.join("result.json");
        fs::write(&report, "preserve existing report").unwrap();
        for invalid in [
            "QualityFirst",
            "ResourceFirst",
            "quality-first",
            "resource-first",
            "quality_first",
            "resource_first",
            "performance",
            "eco",
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
            "unknown",
        ] {
            let output = fixture
                .command_with_profile(false, invalid)
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(1), "{valid}/{invalid:?}");
            assert!(output.stdout.is_empty());
            let envelope = parse_json(std::str::from_utf8(&output.stderr).unwrap()).unwrap();
            assert_eq!(envelope.get("ok"), Some(&JsonValue::Bool(false)));
            let error = envelope.get("error").unwrap();
            assert_eq!(
                error.get("code").and_then(JsonValue::as_str),
                Some("cli_usage")
            );
            assert_eq!(
                error
                    .get("detail")
                    .unwrap()
                    .get("actual")
                    .and_then(JsonValue::as_str),
                Some(invalid)
            );
            assert_eq!(fs::read(&database).unwrap(), before);
            assert_eq!(
                fs::read_to_string(&report).unwrap(),
                "preserve existing report"
            );
            for (path, bytes) in &inputs {
                assert_eq!(&fs::read(path).unwrap(), bytes, "{}", path.display());
            }
        }
    }
}

#[test]
fn benchmark_cli_rejects_resource_profile_fields_in_dataset_and_query_inputs() {
    let fixture = Fixture::new("resource-profile-input-fields", "none");
    fixture.run(false);
    let manifest_path = fixture.0.join("dataset_manifest.json");
    let original_manifest = parse_json(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    let query_path = fixture.0.join("queries.jsonl");
    let original_queries = fs::read_to_string(&query_path).unwrap();
    let (first, rest) = original_queries.split_once('\n').unwrap();
    let database = fixture.0.join("data.coffer");
    let database_bytes = fs::read(&database).unwrap();
    let report = fixture.0.join("result.json");
    fs::write(&report, "preserve existing report").unwrap();

    // SHA-256 values independently computed over each exact query wire below,
    // including both trailing newlines. Never mask schema refusal with a bad hash.
    for (value, profile_hash, resource_profile_hash) in [
        (
            "balanced",
            "29cd3775e74ffd0a45d23e56c6b34ad98414caba1715231d7290bd4db89f6e88",
            "966033e95009e8eed22efb067299cdea7d3da5d33c4fce3010122a27fd108cb1",
        ),
        (
            "QualityFirst",
            "5de7c5b8de1215c92f00a6b8797da3e1c39119a80e83d3c49df9e939d1a5e147",
            "9c352ce647894938e62ec33426d525373accae2e29b1a6e9d74f43980e627464",
        ),
        (
            "ResourceFirst",
            "cae4139a931f114fedd8a49365c431435ffc53c161a1740213d7f30740ba05ee",
            "a8128eaeabc883d9a1843e6e5e2db5d3e566aa90fd7b356cd388dee401645416",
        ),
        (
            "quality-first",
            "e3831d52994593ad92197f8fc2559ca4a0889d7bdc39e8f23370841391e177b1",
            "15fc145955d635a75d0f3f65b9112b56c6370d23f4dc61fad49ae6278788fa8d",
        ),
        (
            "resource-first",
            "ea39f4073b19a47129818b8a4d90d420f896a2caee384af5e758985bee2f7b0e",
            "3ea952b34ea294b85b5f5f04105a2af6fd649ff2780b98b6016def14cc63cfea",
        ),
        (
            "quality_first",
            "092fe8f8f62638760af31c15939aac1eda84d88a0a74763a9e5c4af2bbe4d0aa",
            "983ceb758beb901f3f9e83bd4da41644141b7486617597cbf046121f5a7a0975",
        ),
        (
            "resource_first",
            "639d5b9bf85631017066c18bfea3f9df3b780be307c6deb51511e489aae50db7",
            "14b89acd0f19d0d54831312c6d173c68233b15cb3150f5a8e6be82608fe750d1",
        ),
        (
            "performance",
            "809bd090953d9ee1f216f9236aa17da1562f4428761d63744284d8868398ca2f",
            "cda302ba7aed449f430222f22ff8fd2ffee61d347ec32ce0a41bf403d096bbf7",
        ),
        (
            "eco",
            "639a1f4eb0eb4e316fdd4afd295c402b6e72712ea408688a4ade28907bc17e16",
            "6c8525162f7bf61a608b6d51726f4518f25a20db21a4aa3e83012e419163065a",
        ),
        (
            "Expanded",
            "767c50ffc2cc68b92b0292e0a91525b8d38fd484c28c4ec4eeff13b93cfc137d",
            "43774aefa321d708a8f99172a8688bbe9b58fd5bfcda5920ce73f59eea11ff87",
        ),
        (
            "Balanced",
            "b7b2b12dad3b59ec09511e194063faa0ce389cfb97fac97fd82a9a8c48010c2e",
            "4d3c8e9d42e828152f4646a58a692b64b68d2b3049fbb150c27d7849c8be44c6",
        ),
        (
            "Constrained",
            "be09eb297c1247728fd972f1c07ef312086ed349231e4c87e475df21bff4a5f1",
            "bfacb61656ca905d212f7efb0605687bac8d2089feb1cac18c442bc4b80143d3",
        ),
        (
            "EXPANDED",
            "7f51643d6782dd545962beaee6548bdc1ab25d1f615a9c61d8c1fc0418a7e21f",
            "14d3cf66fb63deee4775ba137ddc60d34316c8848362baf3130a17be9990424b",
        ),
        (
            "BALANCED",
            "536fb88cd1f47827a3d3b1f080120f776a7ec556b9cd1d5869678622f7117d2a",
            "bf7a9e05de97337cd508d3d2a269214a346064df87d305d1ecc8e8c50504f229",
        ),
        (
            "CONSTRAINED",
            "fbdec09ce33baddd319df7334a44e95aee41803d1558ccc34f37283d9c3a2989",
            "796ea020a67de83c2144aacfe88f53f11343e77e55b43cd06aa7ea3fa1347f4a",
        ),
        (
            " expanded",
            "c7ab301bfc76daa1cf2494c2bf4d50c65c537ce8ae5225196fb9cfe6c5d240f6",
            "5a63407c5614a3e2e522d5630f73bf41fb9fe652e2b2a7aff08e98956ec1c3da",
        ),
        (
            "constrained ",
            "8d80d4d6296c118989e02589f6fcda17f3778a5b72a10058a0637909dfd4da89",
            "211338f046f21a0d76a77faf349b1758a5f2cc2221a6d064b78e0f577b5e2a29",
        ),
        (
            "expanded-budget",
            "d95fc89caf3632d44b47fa1ded2ced0038f749c4512a815f5986782d7e9aeda8",
            "0ccb1a6e101a5171f6c4c72c62b6f0a4ae2264155eb023d20667bcfa5b05d7c5",
        ),
        (
            "",
            "7b315c96680070c3cf416a65a5aeab1ba21c886c74d4ae6732171c3c2c6b4730",
            "d8521a7bd99e99d84e0b587c3564697335e8550fdb6d9f88d3abda699b1628ca",
        ),
        (
            "unknown",
            "ea95575cd6b0400e5658c451107b8eaab4805474fd0b3c8d4ab3a98640992da2",
            "7dbe826c6f984989c877c7ee8c62cea31a3278d8deb7c502c362dc7dbaac4a2b",
        ),
        (
            "help",
            "9592186d71b014652bb3abebafcddf513bdfbac5b3f85a4999d82511ff0e8790",
            "a7c141ea6608e4378e59140dd3e6192958dfce54c8b476aefd5205eeedd54c1b",
        ),
    ] {
        for (field, query_hash) in [
            ("profile", profile_hash),
            ("resource_profile", resource_profile_hash),
        ] {
            for location in ["manifest", "queries"] {
                let mut manifest = original_manifest.clone();
                let object = manifest.as_object_mut().unwrap();
                let queries = if location == "manifest" {
                    object.insert(field.into(), JsonValue::String(value.into()));
                    original_queries.clone()
                } else {
                    object
                        .get_mut("hashes")
                        .unwrap()
                        .as_object_mut()
                        .unwrap()
                        .insert(
                            "queries_sha256".into(),
                            JsonValue::String(query_hash.into()),
                        );
                    format!(
                        "{},\"{field}\":\"{value}\"}}\n{rest}",
                        first.strip_suffix('}').unwrap()
                    )
                };
                fs::write(&query_path, queries).unwrap();
                fs::write(&manifest_path, manifest.to_compact_string()).unwrap();
                let inputs: Vec<_> = fs::read_dir(&fixture.0)
                    .unwrap()
                    .map(|entry| {
                        let path = entry.unwrap().path();
                        let bytes = fs::read(&path).unwrap();
                        (path, bytes)
                    })
                    .collect();
                let output = fixture.command(false).output().unwrap();
                assert_eq!(output.status.code(), Some(1), "{location}/{field}/{value}");
                assert!(output.stdout.is_empty());
                let envelope = parse_json(std::str::from_utf8(&output.stderr).unwrap()).unwrap();
                assert_eq!(envelope.get("ok"), Some(&JsonValue::Bool(false)));
                let error = envelope.get("error").unwrap();
                assert_eq!(
                    error.get("code").and_then(JsonValue::as_str),
                    Some("invalid_jsonl_schema")
                );
                assert!(error
                    .get("message")
                    .and_then(JsonValue::as_str)
                    .unwrap()
                    .contains("unsupported field"));
                assert_eq!(
                    error
                        .get("detail")
                        .unwrap()
                        .get("field")
                        .and_then(JsonValue::as_str),
                    Some(location)
                );
                assert_eq!(fs::read(&database).unwrap(), database_bytes);
                assert_eq!(
                    fs::read_to_string(&report).unwrap(),
                    "preserve existing report"
                );
                for (path, bytes) in &inputs {
                    assert_eq!(&fs::read(path).unwrap(), bytes, "{}", path.display());
                }
            }
        }
    }
}

#[test]
fn benchmark_cli_checksum_failures_never_publish_or_mutate_an_existing_database() {
    let fixture = Fixture::new("checksum", "sidecar");
    let database = fixture.0.join("data.coffer");
    Database::create(&database, DatabaseConfig::default())
        .unwrap()
        .close()
        .unwrap();
    let before = fs::read(&database).unwrap();
    for seed in 0..100 {
        let (name, original, hash) = ASSETS[seed % 6];
        let path = fixture.0.join(format!("{name}.jsonl"));
        fs::write(&path, format!("{original}{}", " ".repeat(seed + 1))).unwrap();
        let Output {
            status,
            stderr,
            stdout,
        } = fixture.command(false).output().unwrap();
        assert_eq!(status.code(), Some(2));
        assert!(stdout.is_empty());
        let envelope = parse_json(std::str::from_utf8(&stderr).unwrap()).unwrap();
        let error = envelope.get("error").unwrap();
        assert_eq!(
            error.get("code").and_then(JsonValue::as_str),
            Some("checksum_mismatch")
        );
        let detail = error.get("detail").unwrap();
        assert_eq!(
            detail.get("artifact_kind").and_then(JsonValue::as_str),
            Some(name)
        );
        assert_eq!(
            detail.get("expected_sha256").and_then(JsonValue::as_str),
            Some(hash)
        );
        assert_eq!(fs::read(&database).unwrap(), before);
        assert!(!fixture.0.join("result.json").exists());
        fs::write(path, original).unwrap();
    }
    fixture.run(false);
}

#[test]
fn sidecar_errors_have_typed_details_and_preserve_existing_artifacts() {
    let cases = [
        (
            "duplicate",
            concat!(
                r#"{"query_id":"hit","model_name":"m","dim":2,"vector":[1,0]}"#,
                "\n",
                r#"{"query_id":"hit","model_name":"m","dim":2,"vector":[0,1]}"#,
                "\n"
            ),
            "fd0bd8a5928f4713408161f34b61130a4fd29d671d3ef14d12643d12d98fc0da",
            "invalid_jsonl_schema",
            r#"{"query_id":"hit","first_line":1,"duplicate_line":2}"#,
        ),
        (
            "dim",
            concat!(
                r#"{"query_id":"hit","model_name":"m","dim":3,"vector":[1,0]}"#,
                "\n"
            ),
            "fbf881a5f5222c3540c9408eb68a944d710a8fe48d87ca66b69d69c7eee91937",
            "invalid_vector_length",
            r#"{"query_id":"hit","expected_dim":2,"row_dim":3,"actual_vector_length":2}"#,
        ),
        (
            "length",
            concat!(
                r#"{"query_id":"hit","model_name":"m","dim":2,"vector":[1]}"#,
                "\n"
            ),
            "1ed390e204337ebcb4a7e619aab1e62460e9100decc7344c2bca18e444aad44a",
            "invalid_vector_length",
            r#"{"query_id":"hit","expected_dim":2,"row_dim":2,"actual_vector_length":1}"#,
        ),
        (
            "huge-dim",
            concat!(
                r#"{"query_id":"hit","model_name":"m","dim":1152921504606846976,"vector":[1,0]}"#,
                "\n"
            ),
            "99c15a5ff6764005de6e0d85a0c8c06b015b5d4ef4851e1cbc0a5f9d56208d46",
            "invalid_vector_length",
            r#"{"query_id":"hit","expected_dim":2,"row_dim":1152921504606846976,"actual_vector_length":2}"#,
        ),
        (
            "model",
            concat!(
                r#"{"query_id":"hit","model_name":"other","dim":2,"vector":[1,0]}"#,
                "\n"
            ),
            "3fc3af2b6330f88585408603af0dc0088f9f00c467ca0474cadb3e2e9a9ea762",
            "embedding_model_mismatch",
            r#"{"query_id":"hit","manifest_query_embedding_model":"m","row_model_name":"other"}"#,
        ),
        (
            "missing-model",
            concat!(r#"{"query_id":"hit","dim":2,"vector":[1,0]}"#, "\n"),
            "95a29ccf9eff57d56518de1d6bfb615cb117ecf1d8758f903b3bd64666390997",
            "missing_required_field",
            r#"{"query_id":"hit","field":"model_name"}"#,
        ),
        (
            "bad-dim-type",
            concat!(
                r#"{"query_id":"hit","model_name":"m","dim":"private-dim","vector":[1,0]}"#,
                "\n"
            ),
            "08c0324e8b5afb850f2df91ba09f1c4ff0f42d5b44c4d4140a319145f0c0cbcd",
            "missing_required_field",
            r#"{"query_id":"hit","field":"dim"}"#,
        ),
        (
            "missing-vector",
            concat!(r#"{"query_id":"hit","model_name":"m","dim":2}"#, "\n"),
            "e2e9b952bdd4dd179198b67cab387c96225649c09b417751cafa8db9302dc494",
            "missing_required_field",
            r#"{"query_id":"hit","field":"vector"}"#,
        ),
        (
            "missing",
            concat!(
                r#"{"query_id":"hit","model_name":"m","dim":2,"vector":[1,0]}"#,
                "\n"
            ),
            "32b82f1a8528280f9fcdd2cba53d1713314dee108f6840a817da632511e433d6",
            "missing_required_field",
            r#"{"query_id":"empty","field":"query_embeddings[query_id]","source":"query_embeddings.jsonl"}"#,
        ),
    ];
    for (label, bytes, hash, code, expected_detail) in cases {
        for existing in [false, true] {
            let fixture = Fixture::new(&format!("error-{label}-{existing}"), "sidecar");
            let artifact_path = if existing {
                fs::create_dir(fixture.0.join("nested")).unwrap();
                "nested/query-vectors.jsonl"
            } else {
                "query_embeddings.jsonl"
            };
            fs::write(fixture.0.join(artifact_path), bytes).unwrap();
            let manifest_path = fixture.0.join("dataset_manifest.json");
            let mut manifest = parse_json(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
            manifest.as_object_mut().unwrap().insert(
                "query_embeddings".into(),
                JsonValue::String(artifact_path.into()),
            );
            manifest
                .as_object_mut()
                .unwrap()
                .get_mut("hashes")
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert(
                    "query_embeddings_sha256".into(),
                    JsonValue::String(hash.into()),
                );
            fs::write(manifest_path, manifest.to_compact_string()).unwrap();
            let database = fixture.0.join("data.coffer");
            let result = fixture.0.join("result.json");
            let before = if existing {
                Database::create(&database, DatabaseConfig::default())
                    .unwrap()
                    .close()
                    .unwrap();
                fs::write(&result, b"previous result").unwrap();
                Some(fs::read(&database).unwrap())
            } else {
                None
            };
            let output = fixture.command(false).output().unwrap();
            assert_eq!(output.status.code(), Some(1), "{label}");
            assert!(output.stdout.is_empty());
            let stderr = std::str::from_utf8(&output.stderr).unwrap();
            let envelope = parse_json(stderr).unwrap();
            assert_eq!(envelope.get("ok").and_then(JsonValue::as_bool), Some(false));
            let error = envelope.get("error").unwrap();
            assert_eq!(
                error.get("code").and_then(JsonValue::as_str),
                Some(code),
                "{label}"
            );
            let expected_detail = parse_json(expected_detail).unwrap();
            for (field, expected) in expected_detail.as_object().unwrap() {
                assert_eq!(
                    error.get("detail").unwrap().get(field),
                    Some(expected),
                    "{label}: {field}"
                );
            }
            let detail = error.get("detail").unwrap();
            assert_eq!(
                detail.get("path").and_then(JsonValue::as_str),
                Some(artifact_path),
                "{label}"
            );
            assert_eq!(
                detail.get("artifact_kind").and_then(JsonValue::as_str),
                Some("query_embeddings"),
                "{label}"
            );
            if label != "missing" {
                assert_eq!(
                    detail.get("line").and_then(JsonValue::as_str),
                    Some(if label == "duplicate" { "2" } else { "1" }),
                    "{label}"
                );
            } else {
                assert!(detail.get("line").is_none());
            }
            for private in [
                "[1,0]",
                "[0,1]",
                "recovery",
                "chunk.lang",
                "private-dim",
                fixture.0.to_str().unwrap(),
            ] {
                assert!(!stderr.contains(private), "{label}: leaked {private}");
            }
            if let Some(before) = before {
                assert_eq!(fs::read(database).unwrap(), before);
                assert_eq!(fs::read(result).unwrap(), b"previous result");
            } else {
                assert!(!database.exists());
                assert!(!result.exists());
            }
        }
    }
}

#[test]
fn configured_sqlite_vector_baseline_is_required_in_the_real_cli_process() {
    if std::env::var_os("COFFERDB_SQLITE_VEC_EXTENSION").is_none() {
        eprintln!("external sqlite-vec integration not configured; mandatory in the SQLite CI job");
        return;
    }
    let fixture = Fixture::new("sqlite", "sidecar");
    let result = fixture.run(true);
    assert_eq!(
        result.get("baseline_status").and_then(JsonValue::as_str),
        Some("completed")
    );
    verify_runs(result.get("runs").unwrap(), true);
    verify_runs(result.get("baseline_results").unwrap(), true);
}
