use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cofferdb::{
    parse_json, CheckOptions, ChunkInput, CollectionConfig, CompactOptions, Database,
    DatabaseConfig, DatabaseOpenOptions, DistanceMetric, DocumentInput, Durability, EmbeddingInput,
    ErrorCode, JsonValue, OnConflict, QueryEmbeddingInput, TokenizerConfig, VectorIndexConfig,
};

const WORKER: &str = "COFFERDB_FUZZ_SMOKE_WORKER";
const ROOT: &str = "COFFERDB_FUZZ_SMOKE_ROOT";

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cofferdb-fuzz-smoke-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn run_worker(test: &str, mode: &str, root: &Path, timeout: Duration) -> Option<ExitStatus> {
    let log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("worker.log"))
        .unwrap();
    let mut worker = Worker(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env(WORKER, mode)
            .env(ROOT, root)
            .stdout(Stdio::from(log.try_clone().unwrap()))
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap(),
    );
    let started = Instant::now();
    loop {
        if let Some(status) = worker.0.try_wait().unwrap() {
            return Some(status);
        }
        if started.elapsed() >= timeout {
            return None;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn isolated(test: &str, body: impl FnOnce(&Path)) {
    if std::env::var(WORKER).as_deref() == Ok(test) {
        body(Path::new(&std::env::var_os(ROOT).unwrap()));
        return;
    }
    let fixture = Fixture::new();
    let status = run_worker(test, test, &fixture.0, Duration::from_secs(45));
    assert!(
        status.is_some_and(|status| status.success()),
        "worker {test}: {status:?} (None means timeout)\n{}",
        fs::read_to_string(fixture.0.join("worker.log")).unwrap()
    );
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, limit: usize) -> usize {
        (self.next() % limit as u64) as usize
    }
}

fn mutate(input: &[u8], random: &mut Random) -> Vec<u8> {
    let mut bytes = input.to_vec();
    match random.below(4) {
        0 if !bytes.is_empty() => {
            let position = random.below(bytes.len());
            bytes[position] ^= 1 << random.below(8);
        }
        1 if !bytes.is_empty() => {
            bytes.remove(random.below(bytes.len()));
        }
        2 => bytes.truncate(random.below(bytes.len() + 1)),
        _ => bytes.insert(random.below(bytes.len() + 1), random.next() as u8),
    }
    bytes
}

#[test]
fn json_fixed_seed_mutations_preserve_roundtrip_or_return_errors() {
    isolated(
        "json_fixed_seed_mutations_preserve_roundtrip_or_return_errors",
        |_| {
            let mut corpus: Vec<String> = [
                "null",
                "true",
                "-0.0",
                "18446744073709551615",
                "-9223372036854775808",
                "1e309",
                "[0,1.5,-2.75,1e-300]",
                r#"{"a":1,"a":2}"#,
                r#"{"nested":[true,null,{"a":"\u0000\uD83D\uDE00"}]}"#,
                r#""\uD800""#,
                r#""\uDC00""#,
                "\"\u{e9}\u{4e2d}\u{1f600}\"",
                "[",
                "{\"key\":",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect();
            for depth in [127, 128, 129] {
                corpus.push(format!("{}0{}", "[".repeat(depth), "]".repeat(depth)));
                corpus.push(format!(
                    "{}null{}",
                    "{\"v\":".repeat(depth),
                    "}".repeat(depth)
                ));
            }
            let mut accepted = 0;
            let mut rejected = 0;
            let mut non_utf8 = 0;
            for seed in 1..=100 {
                let mut random = Random(seed);
                for (case, original) in corpus.iter().enumerate() {
                    for mutation in 0..=4 {
                        eprintln!("json seed={seed} case={case} mutation={mutation}");
                        let bytes = if mutation == 0 {
                            original.as_bytes().to_vec()
                        } else {
                            mutate(original.as_bytes(), &mut random)
                        };
                        let Ok(input) = std::str::from_utf8(&bytes) else {
                            non_utf8 += 1;
                            continue;
                        };
                        match parse_json(input) {
                            Ok(value) => {
                                accepted += 1;
                                let canonical = value.try_to_compact_string().unwrap();
                                let reparsed = parse_json(&canonical).unwrap();
                                assert_eq!(value, reparsed);
                                assert_eq!(canonical, reparsed.try_to_compact_string().unwrap());
                                assert_eq!(value, parse_json(&value.to_pretty_string()).unwrap());
                                assert_eq!(value, value.clone());
                            }
                            Err(_) => rejected += 1,
                        }
                    }
                }
            }
            assert!(accepted > 0 && rejected > 0 && non_utf8 > 0);
        },
    );
}

fn assert_read_only(path: &Path, bytes: &[u8], preserve: bool) {
    fs::write(path, bytes).unwrap();
    let opened = Database::open_read_only(path);
    if preserve {
        assert!(opened.is_ok(), "index-only damage: {:?}", opened.err());
    }
    if let Ok(db) = opened {
        let stats = db.stats().unwrap();
        assert!(stats.collection_count <= 1);
        assert!(stats.document_count <= 1);
        assert!(stats.chunk_count <= 1);
        assert!(stats.embedding_count <= 1);
        if preserve {
            assert_eq!(stats.document_count, 1);
            assert_eq!(stats.chunk_count, 1);
            assert_eq!(stats.embedding_count, 1);
            let hits = db
                .collection("notes")
                .unwrap()
                .search()
                .text("recovery")
                .execute()
                .unwrap();
            assert_eq!(hits.results.len(), 1);
            assert_eq!(hits.results[0].chunk_external_id, "doc#1");
            let vectors = db
                .collection("notes")
                .unwrap()
                .search()
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("fixture".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .execute()
                .unwrap();
            assert_eq!(vectors.results.len(), 1);
            assert_eq!(vectors.results[0].chunk_external_id, "doc#1");
            assert_eq!(vectors.results[0].vector_score, Some(1.0));
        }
    }
    let checked = Database::check_path(
        path,
        CheckOptions {
            verify_indexes: true,
        },
    );
    if preserve {
        let report = checked.unwrap();
        assert_eq!(report.live_documents, 1);
        assert_eq!(report.live_chunks, 1);
    }
    assert_eq!(fs::read(path).unwrap(), bytes);
}

fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82f6_3b78 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

#[test]
fn storage_fixed_seed_damage_never_changes_read_only_input() {
    isolated(
        "storage_fixed_seed_damage_never_changes_read_only_input",
        |root| {
            let baseline_path = root.join("baseline.coffer");
            let db = Database::create(&baseline_path, DatabaseConfig::default()).unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    tokenizer: TokenizerConfig::default_en(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                })
                .unwrap();
            collection
                .insert_document(DocumentInput {
                    external_id: "doc".into(),
                    title: None,
                    source_uri: None,
                    metadata: parse_json(r#"{"value":[1,true,null]}"#).unwrap(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: "doc#1".into(),
                        text: "recovery storage".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
            collection
                .put_embeddings(vec![EmbeddingInput {
                    chunk_external_id: "doc#1".into(),
                    model_name: "fixture".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                }])
                .unwrap();
            drop(collection);
            drop(db);
            let baseline = fs::read(&baseline_path).unwrap();
            let report = Database::check_path(&baseline_path, CheckOptions::default()).unwrap();
            assert_eq!(report.status, "ok");
            let critical_end = report.critical_end_offset as usize;
            assert!(critical_end < baseline.len());
            let path = root.join("mutated.coffer");
            assert_read_only(&path, &baseline, true);
            for length in 0..baseline.len() {
                eprintln!("storage truncate={length}");
                assert_read_only(&path, &baseline[..length], length >= critical_end);
            }
            for seed in 1..=100 {
                let mut random = Random(seed);
                for case in 0..8 {
                    eprintln!("storage seed={seed} mutation={case}");
                    assert_read_only(&path, &mutate(&baseline, &mut random), false);
                }
                let mut suffix = baseline.clone();
                let offset = critical_end + random.below(baseline.len() - critical_end);
                suffix[offset] ^= 1 << random.below(8);
                eprintln!("storage seed={seed} suffix_offset={offset}");
                assert_read_only(&path, &suffix, true);
            }
            assert_eq!(crc32c(b"123456789"), 0xe306_9283);
            for trailing in [&b"CFRL"[..], &b"CFIDXCHK"[..], &[0xff; 100][..]] {
                eprintln!("storage trailing={trailing:?}");
                let mut bytes = baseline.clone();
                bytes.extend_from_slice(trailing);
                assert_read_only(&path, &bytes, true);
            }
            for payload_len in [0, 1, u32::MAX as u64, u64::MAX - 50, u64::MAX] {
                eprintln!("storage forged_payload_len={payload_len}");
                let mut forged = baseline.clone();
                let header = &mut forged[58..108];
                assert_eq!(&header[..4], b"CFRL");
                header[8..16].copy_from_slice(&payload_len.saturating_add(50).to_le_bytes());
                header[34..42].copy_from_slice(&payload_len.to_le_bytes());
                let crc = crc32c(&header[..46]);
                header[46..50].copy_from_slice(&crc.to_le_bytes());
                assert_read_only(&path, &forged, false);
            }
            assert_eq!(fs::read(baseline_path).unwrap(), baseline);
        },
    );
}

#[test]
fn worker_guard_detects_failure_and_reaps_timeout() {
    match std::env::var(WORKER).as_deref() {
        Ok("sleep") => {
            std::thread::sleep(Duration::from_secs(60));
            return;
        }
        Ok("fail") => panic!("intentional worker failure"),
        _ => {}
    }
    for mode in ["sleep", "fail"] {
        let fixture = Fixture::new();
        let timeout = if mode == "sleep" {
            Duration::from_millis(100)
        } else {
            Duration::from_secs(10)
        };
        let status = run_worker(
            "worker_guard_detects_failure_and_reaps_timeout",
            mode,
            &fixture.0,
            timeout,
        );
        if mode == "sleep" {
            assert!(status.is_none());
        } else {
            assert!(!status.unwrap().success());
            assert!(fs::read_to_string(fixture.0.join("worker.log"))
                .unwrap()
                .contains("intentional worker failure"));
        }
    }
}

#[derive(Clone)]
struct ModelDocument {
    revision: usize,
    // The value is the expected cosine score against [1, 0], not an engine result.
    chunks: BTreeMap<usize, Option<f32>>,
}

type CollectionModel = BTreeMap<usize, ModelDocument>;

fn model_config(name: &str) -> CollectionConfig {
    CollectionConfig {
        name: name.into(),
        tokenizer: TokenizerConfig::default_en(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: Some("model".into()),
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    }
}

fn model_document(id: usize, revision: usize) -> DocumentInput {
    DocumentInput {
        external_id: format!("doc-{id}"),
        title: None,
        source_uri: None,
        metadata: parse_json(&format!(r#"{{"revision":{revision}}}"#)).unwrap(),
        chunks: (0..3)
            .map(|ordinal| ChunkInput {
                chunk_external_id: format!("doc-{id}#{ordinal}"),
                text: format!("shared revision {revision}"),
                ordinal,
                metadata: JsonValue::object(),
                source_span: None,
            })
            .collect(),
    }
}

fn assert_model(db: &Database, models: &[CollectionModel; 2]) {
    let mut documents = 0;
    let mut chunks = 0;
    let mut embeddings = 0;
    for (name, model) in ["left", "right"].into_iter().zip(models) {
        let collection = db.collection(name).unwrap();
        let mut expected_text = BTreeSet::new();
        let mut expected_vectors = BTreeMap::new();
        for (&id, expected) in model {
            for (&ordinal, score) in &expected.chunks {
                let external_id = format!("doc-{id}#{ordinal}");
                expected_text.insert(external_id.clone());
                if let Some(score) = score {
                    expected_vectors.insert(external_id, *score as f64);
                }
            }
            let doc = collection
                .get_document_by_external_id(&format!("doc-{id}"))
                .unwrap();
            assert_eq!(
                doc.metadata.get("revision").unwrap().as_f64(),
                Some(expected.revision as f64)
            );
            let actual = collection.list_chunks(&doc.document_id).unwrap();
            assert_eq!(actual.len(), expected.chunks.len());
            for chunk in actual {
                let vector = expected.chunks.get(&chunk.ordinal).unwrap();
                let external_id = format!("doc-{id}#{}", chunk.ordinal);
                assert_eq!(chunk.chunk_external_id, external_id);
                assert_eq!(chunk.text, format!("shared revision {}", expected.revision));
                assert_eq!(chunk.embedding_id.is_some(), vector.is_some());
            }
        }
        let stats = collection.stats().unwrap();
        assert_eq!(stats.document_count, model.len());
        assert_eq!(stats.chunk_count, expected_text.len());
        assert_eq!(stats.embedding_count, expected_vectors.len());
        documents += model.len();
        chunks += expected_text.len();
        embeddings += expected_vectors.len();
        for mode in 0..3 {
            let mut query = collection.search().top_k(100);
            if mode != 1 {
                query = query.text("shared");
            }
            if mode != 0 {
                query = query.query_embedding(QueryEmbeddingInput {
                    model_name: Some("model".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                });
            }
            let result = query.execute().unwrap();
            let ids = result
                .results
                .iter()
                .map(|hit| hit.chunk_external_id.clone())
                .collect::<BTreeSet<_>>();
            assert_eq!(ids.len(), result.results.len());
            assert_eq!(
                ids,
                if mode == 1 {
                    expected_vectors.keys().cloned().collect()
                } else {
                    expected_text.clone()
                }
            );
            for hit in result.results {
                assert_eq!(
                    hit.vector_score,
                    if mode == 0 {
                        None
                    } else {
                        expected_vectors.get(&hit.chunk_external_id).copied()
                    }
                );
                assert_eq!(hit.bm25_score.is_some(), mode != 1);
            }
        }
    }
    let stats = db.stats().unwrap();
    assert_eq!(stats.collection_count, 2);
    assert_eq!(stats.document_count, documents);
    assert_eq!(stats.chunk_count, chunks);
    assert_eq!(stats.embedding_count, embeddings);
}

#[test]
fn fixed_seed_lifecycle_matches_independent_visibility_model() {
    isolated(
        "fixed_seed_lifecycle_matches_independent_visibility_model",
        |root| {
            let mut operations = [0; 7];
            let mut compact_modes = [0; 2];
            for seed in 1..=12 {
                let directory = root.join(format!("seed-{seed}"));
                fs::create_dir(&directory).unwrap();
                let path = directory.join("data.coffer");
                // This checks visibility and snapshot ownership; crash durability has separate tests.
                let mut db = Database::create(
                    &path,
                    DatabaseConfig {
                        durability: Durability::Fast,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut models: [CollectionModel; 2] = Default::default();
                for (name, model) in ["left", "right"].into_iter().zip(&mut models) {
                    db.create_collection(model_config(name))
                        .unwrap()
                        .insert_document(model_document(0, 0))
                        .unwrap();
                    model.insert(
                        0,
                        ModelDocument {
                            revision: 0,
                            chunks: (0..3).map(|i| (i, None)).collect(),
                        },
                    );
                }
                let old = Database::open_read_only(&path).unwrap();
                let old_models = models.clone();
                let mut random = Random(seed);
                for step in 0..18 {
                    let owner = random.below(2);
                    let name = ["left", "right"][owner];
                    let operation = (step + seed as usize) % 6;
                    eprintln!(
                        "lifecycle seed={seed} step={step} collection={name} operation={operation}"
                    );
                    if operation == 0 || models[owner].is_empty() {
                        operations[0] += 1;
                        let id = step + 1;
                        db.collection(name)
                            .unwrap()
                            .insert_document(model_document(id, step + 1))
                            .unwrap();
                        models[owner].insert(
                            id,
                            ModelDocument {
                                revision: step + 1,
                                chunks: (0..3).map(|i| (i, None)).collect(),
                            },
                        );
                    } else {
                        let id = *models[owner]
                            .keys()
                            .nth(random.below(models[owner].len()))
                            .unwrap();
                        match operation {
                            1 => {
                                if let Some(&ordinal) = models[owner][&id].chunks.keys().next() {
                                    operations[1] += 1;
                                    let collection = db.collection(name).unwrap();
                                    let chunk = collection
                                        .get_chunk_by_external_id(&format!("doc-{id}#{ordinal}"))
                                        .unwrap();
                                    collection.delete_chunk(&chunk.chunk_id).unwrap();
                                    models[owner].get_mut(&id).unwrap().chunks.remove(&ordinal);
                                    assert_eq!(
                                        collection.get_chunk(&chunk.chunk_id).unwrap_err().code(),
                                        ErrorCode::ChunkNotFound
                                    );
                                }
                            }
                            2 => {
                                operations[2] += 1;
                                let collection = db.collection(name).unwrap();
                                let doc = collection
                                    .get_document_by_external_id(&format!("doc-{id}"))
                                    .unwrap();
                                collection.delete_document(&doc.document_id).unwrap();
                                models[owner].remove(&id);
                                assert_eq!(
                                    collection
                                        .get_document(&doc.document_id)
                                        .unwrap_err()
                                        .code(),
                                    ErrorCode::DocumentNotFound
                                );
                            }
                            3 => {
                                if let Some(&ordinal) = models[owner][&id].chunks.keys().next() {
                                    let score = [-1.0, 0.0, 1.0][random.below(3)];
                                    let embedding = EmbeddingInput {
                                        chunk_external_id: format!("doc-{id}#{ordinal}"),
                                        model_name: "model".into(),
                                        dim: 2,
                                        vector: vec![score, if score == 0.0 { 1.0 } else { 0.0 }],
                                    };
                                    let before = fs::read(&path).unwrap();
                                    let result = db
                                        .collection(name)
                                        .unwrap()
                                        .put_embeddings(vec![embedding]);
                                    let expected = models[owner]
                                        .get_mut(&id)
                                        .unwrap()
                                        .chunks
                                        .get_mut(&ordinal)
                                        .unwrap();
                                    if expected.is_some() {
                                        operations[6] += 1;
                                        assert_eq!(
                                            result.unwrap_err().code(),
                                            ErrorCode::DuplicateExternalId
                                        );
                                        assert_eq!(fs::read(&path).unwrap(), before);
                                    } else {
                                        operations[3] += 1;
                                        result.unwrap();
                                        *expected = Some(score);
                                    }
                                }
                            }
                            4 => {
                                operations[4] += 1;
                                let rebuild_indexes = (step / 6 + seed as usize) % 2 == 0;
                                compact_modes[usize::from(rebuild_indexes)] += 1;
                                db.compact(CompactOptions { rebuild_indexes }).unwrap();
                            }
                            5 => {
                                operations[5] += 1;
                                let revision = 100 + step;
                                fs::write(directory.join("documents.jsonl"), format!("{{\"external_id\":\"doc-{id}\",\"metadata\":{{\"revision\":{revision}}}}}\n")).unwrap();
                                fs::write(directory.join("chunks.jsonl"), format!("{{\"document_external_id\":\"doc-{id}\",\"chunk_external_id\":\"doc-{id}#0\",\"ordinal\":0,\"text\":\"shared revision {revision}\",\"metadata\":{{}}}}\n")).unwrap();
                                let manifest = directory.join("manifest.json");
                                fs::write(&manifest, format!(r#"{{"schema_version":"0.1","collection":"{name}","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"embedding_model":"model","distance_metric":"cosine"}}"#)).unwrap();
                                db.import_manifest(manifest, OnConflict::Replace).unwrap();
                                models[owner].insert(
                                    id,
                                    ModelDocument {
                                        revision,
                                        chunks: BTreeMap::from([(0, None)]),
                                    },
                                );
                            }
                            _ => unreachable!(),
                        }
                    }
                    assert_model(&db, &models);
                    assert_model(&old, &old_models);
                    if step % 3 == 2 {
                        db.close().unwrap();
                        db = Database::open_with_options(
                            &path,
                            DatabaseOpenOptions {
                                durability: Some(Durability::Fast),
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        assert_model(&db, &models);
                    }
                }
                db.rebuild_index(None).unwrap();
                assert!(db.check(CheckOptions::default()).unwrap().issues.is_empty());
                let copy =
                    Database::create(directory.join("reimport.coffer"), DatabaseConfig::default())
                        .unwrap();
                for name in ["left", "right"] {
                    let output = directory.join(name);
                    db.export_collection_jsonl(name, &output).unwrap();
                    copy.import_manifest(output.join("export_manifest.json"), OnConflict::Error)
                        .unwrap();
                }
                assert_model(&copy, &models);
                assert_model(&old, &old_models);
            }
            eprintln!("lifecycle operations={operations:?} compact_modes={compact_modes:?}");
            assert!(operations.into_iter().all(|count| count > 0));
            assert!(compact_modes.into_iter().all(|count| count > 0));
        },
    );
}
